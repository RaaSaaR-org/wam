"""Tests for ``scripts/build_identity_prompt_sheet.py`` — the T40-TODO-01 evidence harness.

The happy path is one test out of many here, and deliberately so. This harness runs ONCE in the
life of the experiment, before 4 020 arm-C clips are generated, and every failure it exists to
prevent produces a file that reads like a finished record: a sample that clusters in one recording
session, a verdict averaged over the rows somebody bothered to fill, a count attached to a prompt
that has since been reworded, five rows deleted so the sample shrinks past the refusal that only
knows how to see a blank, a frame taken at the one instant the rubric calls unjudgeable while the
record still says "early", a ``gate_qualified: true`` inferred from a key that is simply absent.
None of those crash. So the tests below inject each of them.

Nothing here needs a GPU, a network, Isaac, a model weight or ffmpeg: the manifest is synthetic and
built in ``tmp_path``, and frame extraction is stubbed at the module boundary. The one thing the
tests DO read from the repository is the committed ``configs/transfer25/styles.toml``, because the
prompt under test is the committed one — a fixture copy would let this suite pass against a string
arm C is not generated from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_identity_prompt_sheet as bips  # noqa: E402

STYLES = REPO / "configs" / "transfer25" / "styles.toml"

# Bigger than the 40-episode default sample so the strata are non-degenerate, and NOT 402, so the
# "this manifest is not the committed corpus" disqualification is exercised by default rather than
# needing its own fixture.
N_SYNTHETIC_EPISODES = 120


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A synthetic ``pr08-apple-640x480`` tree: a manifest and one empty file per episode.

    The videos are empty because nothing in these tests decodes one — ``extract_frame`` is stubbed.
    They must EXIST, though, because build-sheet refuses a manifest whose files are not there, and
    that refusal is itself under test.
    """
    root = tmp_path / "source"
    (root / "videos").mkdir(parents=True)
    episodes = []
    for i in range(N_SYNTHETIC_EPISODES):
        rel = f"videos/episode_{i:06d}.mp4"
        (root / rel).write_bytes(b"")
        episodes.append({"id": f"episode_{i:06d}", "frames": 300 + i, "video": rel})
    (root / "manifest.json").write_text(json.dumps({
        "resolution": [640, 480],
        "fps": 30,
        "video_key": "observation.images.ego_view",
        "episodes": episodes,
    }))
    return root


@pytest.fixture
def stub_frames(monkeypatch: pytest.MonkeyPatch):
    """Replace ffmpeg with a writer of deterministic bytes. No decoder is involved anywhere."""
    def fake(video: Path, index: int, out: Path, ffmpeg: str = "ffmpeg") -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f"PNG:{video.name}:{index}".encode())

    monkeypatch.setattr(bips, "extract_frame", fake)
    return fake


def run(*argv: str) -> int:
    return bips.main(list(argv))


def build(corpus: Path, out: Path, *extra: str) -> int:
    return run("build-sheet", "--manifest", str(corpus / "manifest.json"),
               "--styles", str(STYLES), "--out", str(out), *extra)


def rows_of(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out / "sheet.jsonl").read_text().splitlines() if ln.strip()]


def write_rows(out: Path, rows: list[dict]) -> None:
    (out / "sheet.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def fill(out: Path, verdicts: dict[str, str], axes: dict[str, list[str]] | None = None,
         notes: dict[str, str] | None = None) -> None:
    """Write answers into an existing sheet the way a human or a VLM would."""
    axes = axes or {}
    notes = notes or {}
    rows = rows_of(out)
    for r in rows:
        r["verdict"] = verdicts.get(r["episode"], "")
        r["mismatched_axes"] = axes.get(r["episode"], [])
        r["notes"] = notes.get(r["episode"], "")
    write_rows(out, rows)


def build_and_fill(corpus: Path, out: Path, *extra: str) -> list[dict]:
    """A complete, legally filled 40-row sheet — the state every tampering test starts from."""
    build(corpus, out, "--sample-size", "40", *extra)
    fill(out, {r["episode"]: "match" for r in rows_of(out)})
    return rows_of(out)


def verdict_of(out: Path, art: Path) -> int:
    return run("verdict", "--sheet", str(out), "--styles", str(STYLES), "--out", str(art))


def meta_of(out: Path) -> dict:
    return json.loads((out / "sheet_meta.json").read_text())


def write_meta(out: Path, meta: dict) -> None:
    (out / "sheet_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


# ------------------------------------------------------------------------------------------------
# sampling: deterministic, and spanning
# ------------------------------------------------------------------------------------------------


def test_sampling_is_deterministic_in_the_seed():
    ids = [f"episode_{i:06d}" for i in range(402)]
    a = bips.sample_episode_ids(ids, 40, 40001)
    b = bips.sample_episode_ids(ids, 40, 40001)
    assert a == b
    assert len(a) == 40


def test_sampling_is_independent_of_manifest_order():
    """The id is the recording index; the file's order must not be able to change the sample.

    A manifest rewritten in another order (a rebuild, a resume, a shuffled work list) would
    otherwise produce different evidence under an unchanged, recorded seed.
    """
    ids = [f"episode_{i:06d}" for i in range(402)]
    shuffled = list(reversed(ids))
    assert bips.sample_episode_ids(shuffled, 40, 40001) == bips.sample_episode_ids(ids, 40, 40001)


def test_a_different_seed_draws_a_different_sample():
    ids = [f"episode_{i:06d}" for i in range(402)]
    draws = {tuple(bips.sample_episode_ids(ids, 40, s)) for s in (40001, 40002, 40003, 40004)}
    assert len(draws) > 1, "the seed is not reaching the draw at all"


def test_sample_spans_the_corpus_one_per_stratum():
    """The property the TODO's 'spanning the corpus' asks for, asserted rather than eyeballed.

    One episode from each of the 40 contiguous strata means no gap of more than two strata can
    exist anywhere in the recording order — which is what rules out a sample that accidentally
    covers a single recording session and answers a different question.
    """
    ids = [f"episode_{i:06d}" for i in range(402)]
    picks = bips.sample_episode_ids(ids, 40, 40001)
    idx = [ids.index(p) for p in picks]

    assert idx == sorted(idx)
    assert len(set(idx)) == 40, "an episode was drawn twice"
    for k, (lo, hi) in enumerate(bips.strata_bounds(402, 40)):
        assert lo <= idx[k] < hi, f"stratum {k} was drawn from outside its own range"
    # Every stratum is ~10 episodes wide, so consecutive picks are at most two strata apart.
    assert max(b - a for a, b in zip(idx, idx[1:])) <= 2 * (402 // 40) + 2


def test_strata_cover_every_episode_and_leave_none_empty():
    for m, n in ((402, 40), (402, 7), (120, 40), (41, 40), (40, 40)):
        bounds = bips.strata_bounds(m, n)
        assert len(bounds) == n
        assert bounds[0][0] == 0 and bounds[-1][1] == m
        assert all(lo < hi for lo, hi in bounds), "an empty stratum silently shrinks the sample"
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(n - 1))


def test_sample_larger_than_corpus_is_refused():
    with pytest.raises(bips.SheetError, match="repeat an episode"):
        bips.sample_episode_ids([f"episode_{i:06d}" for i in range(5)], 40, 40001)


# ------------------------------------------------------------------------------------------------
# build-sheet
# ------------------------------------------------------------------------------------------------


def test_build_sheet_writes_blank_rows_carrying_the_committed_prompt(corpus, tmp_path,
                                                                    stub_frames):
    out = tmp_path / "sheet"
    rc = build(corpus, out)

    rows = rows_of(out)
    assert len(rows) == bips.DEFAULT_SAMPLE_SIZE
    assert all(r["verdict"] == "" for r in rows), "a sheet that arrives judged is not a sheet"
    assert all(r["mismatched_axes"] == [] and r["notes"] == "" for r in rows)

    committed = bips.read_identity_style(STYLES)["prompt"]
    assert all(r["prompt"] == committed for r in rows)
    assert all(Path(r["frame"]).is_file() for r in rows)
    # 0.10 of the way in, resolved per clip against that clip's own length.
    assert all(r["frame_index"] == round(0.10 * (r["n_frames"] - 1)) for r in rows)

    meta = json.loads((out / "sheet_meta.json").read_text())
    assert meta["sample_seed"] == bips.DEFAULT_SAMPLE_SEED
    assert meta["sample_size"] == bips.DEFAULT_SAMPLE_SIZE
    assert meta["sampled_episodes"] == [r["episode"] for r in rows]
    assert meta["judge"] is None, "this script must not choose a judge"

    # 120 synthetic episodes is not the committed 402, so the sheet is stamped and exits 3.
    assert rc == bips.EXIT_NOT_GATE_QUALIFIED
    assert any("402" in r for r in meta["gate_disqualified_reasons"])


def test_build_sheet_names_no_judge_anywhere():
    """The T-041 failure mode, guarded structurally rather than by intention.

    T-041's VOID came from a VLM judge that answered a constant 'NO' to all 80 items and could not
    clear its own 20/20 calibration set. This harness therefore names no model, no inference
    library and no transport; if any of them ever appears, this test is the thing that has to be
    argued with first.

    What this CAN and CANNOT see, stated so it is not over-read: it is a token list over the
    source, so it catches the ways a judge is normally reached — an HTTP client, an inference
    library, a vendor SDK, a weights loader, a raw socket — and it does not catch a judge invoked
    through a name nobody thought of, or shelled out through the ``subprocess`` this file already
    imports for ffmpeg. It is a tripwire on the obvious paths, not a proof of absence; the proof is
    that ``verdict`` reads a field and applies a fixed rule to whatever is in it.
    """
    src = (REPO / "scripts" / "build_identity_prompt_sheet.py").read_text().lower()
    forbidden = (
        # transports
        "urllib", "requests", "httpx", "aiohttp", "socket", "websocket", "http://", "https://",
        # vendor SDKs and their endpoints
        "openai", "anthropic", "chat/completions", "api_key", "bearer ", "vertexai", "boto3",
        # local inference and weights
        "transformers", "from_pretrained", "automodel", "torch", "onnxruntime", "llama_cpp",
        "vllm", "ollama", "safetensors",
    )
    for token in forbidden:
        assert token not in src, f"a judge crept in via {token!r}"


def test_skip_frames_disqualifies_but_still_writes(corpus, tmp_path):
    out = tmp_path / "sheet"
    rc = build(corpus, out, "--skip-frames")
    assert rc == bips.EXIT_NOT_GATE_QUALIFIED
    rows = rows_of(out)
    assert rows and all(r["frame"] is None for r in rows)
    meta = json.loads((out / "sheet_meta.json").read_text())
    assert not meta["gate_qualified"]
    assert any("--skip-frames" in r for r in meta["gate_disqualified_reasons"])


def test_small_sample_is_stamped_as_a_smoke_run(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    rc = build(corpus, out, "--sample-size", "5")
    assert rc == bips.EXIT_NOT_GATE_QUALIFIED
    meta = json.loads((out / "sheet_meta.json").read_text())
    assert any("--sample-size 5" in r for r in meta["gate_disqualified_reasons"])
    assert len(rows_of(out)) == 5


def test_missing_video_is_refused_not_sampled_around(corpus, tmp_path, stub_frames):
    for p in (corpus / "videos").glob("*.mp4"):
        p.unlink()
    assert build(corpus, tmp_path / "sheet") == bips.EXIT_FATAL


def test_wrong_resolution_manifest_is_refused(corpus, tmp_path, stub_frames):
    """The 120x160 converted corpus has a valid manifest and is the corpus PR-08 §3 forbids."""
    man = json.loads((corpus / "manifest.json").read_text())
    man["resolution"] = [160, 120]
    (corpus / "manifest.json").write_text(json.dumps(man))
    assert build(corpus, tmp_path / "sheet") == bips.EXIT_FATAL


def test_reseeding_over_a_sheet_is_refused_without_the_flag(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    build(corpus, out)
    assert build(corpus, out, "--seed", "999") == bips.EXIT_FATAL
    assert build(corpus, out, "--seed", "999", "--allow-reseed") in (
        bips.EXIT_OK, bips.EXIT_NOT_GATE_QUALIFIED)
    assert json.loads((out / "sheet_meta.json").read_text())["sample_seed"] == 999


def test_a_filled_sheet_is_never_overwritten_by_a_rebuild(corpus, tmp_path, stub_frames, capsys):
    """--allow-reseed's message says 'if the first draw was never filled'. That is now checked.

    Two ways to lose forty verdicts silently: reseed over them (sample-shopping, and the flag was
    documented as covering only an unfilled draw), or simply re-run the identical command, whose
    sheet_id matches and which therefore sailed through the reseed guard and wrote forty blank rows
    over the answers. Neither is a wrong number, but both destroy the only record of what somebody
    saw in forty frames.
    """
    out = tmp_path / "sheet"
    build_and_fill(corpus, out)

    assert build(corpus, out, "--sample-size", "40") == bips.EXIT_FATAL
    assert build(corpus, out, "--sample-size", "40", "--seed", "999",
                 "--allow-reseed") == bips.EXIT_FATAL
    assert "already carries 40 filled verdict(s)" in capsys.readouterr().err
    assert all(r["verdict"] == "match" for r in rows_of(out)), "the answers were overwritten"


def test_reseeding_clears_the_previous_draws_frames(corpus, tmp_path, stub_frames):
    """Frames of episodes this sheet does not name must not sit next to the ones under judgement."""
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    before = {p.name for p in (out / "frames").glob("*.png")}

    build(corpus, out, "--sample-size", "40", "--seed", "999", "--allow-reseed")
    after = {p.name for p in (out / "frames").glob("*.png")}
    drawn = {f"{r['episode']}.png" for r in rows_of(out)}

    assert after == drawn, "a frame from a draw this sheet does not name was left behind"
    assert before - after, "the two draws are identical; the fixture is not exercising this"
    assert meta_of(out)["stale_frames_removed"] == sorted(before - after)


def test_build_sheet_refuses_a_jsonl_out(corpus, tmp_path, stub_frames):
    """--out foo.jsonl put the rows and the frames/ directory at the same path.

    It used to extract all forty frames and then die on IsADirectoryError writing the rows: a path
    the code cannot honour, accepted, acted on, and abandoned half-done.
    """
    out = tmp_path / "sheet.jsonl"
    assert build(corpus, out) == bips.EXIT_FATAL
    assert not out.exists()


# ------------------------------------------------------------------------------------------------
# the frame fraction: a flag with a range, and a rule the record has to match
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["9.0", "-0.1", "1.5", "nan"])
def test_frame_fraction_outside_the_clip_is_refused_not_clamped(corpus, tmp_path, stub_frames,
                                                                bad):
    """Clamped, --frame-fraction 9 takes the LAST frame of every clip and says it took an early one.

    That is the shape this whole track is about: no crash, forty extracted frames, a gate-qualified
    artifact, and every verdict about the one instant the rubric names as unjudgeable (the apple on
    the plate, the hand over the region the prompt describes).
    """
    out = tmp_path / "sheet"
    assert build(corpus, out, "--frame-fraction", bad) == bips.EXIT_FATAL
    assert not (out / "sheet.jsonl").exists(), "a refusal must not leave a sheet behind"


def test_a_late_frame_fraction_runs_but_is_stamped_and_says_so(corpus, tmp_path, stub_frames):
    """In range but not the pre-registered rule: produced, recorded honestly, not gate-qualified."""
    out = tmp_path / "sheet"
    rc = build(corpus, out, "--sample-size", "40", "--frame-fraction", "0.9")
    assert rc == bips.EXIT_NOT_GATE_QUALIFIED

    meta = meta_of(out)
    assert any("0.9" in r and "0.25" in r for r in meta["gate_disqualified_reasons"])
    # The recorded rule must describe the run, not the default.
    assert "Early rather than middle or late" not in meta["frame_rule"]
    assert "NOT the pre-registered rule" in meta["frame_rule"]
    rows = rows_of(out)
    assert all(r["frame_index"] == round(0.9 * (r["n_frames"] - 1)) for r in rows)


def test_frame_fraction_zero_is_the_frame_the_docstring_rules_out(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    assert build(corpus, out, "--sample-size", "40",
                 "--frame-fraction", "0") == bips.EXIT_NOT_GATE_QUALIFIED
    meta = meta_of(out)
    assert any("frame 0" in r for r in meta["gate_disqualified_reasons"])
    assert all(r["frame_index"] == 0 for r in rows_of(out))


def test_the_default_fraction_is_gate_qualifiable(corpus, tmp_path, stub_frames):
    """The stamp above must not fire on the rule the harness was designed around."""
    assert bips.frame_fraction_disqualification(bips.DEFAULT_FRAME_FRACTION) is None
    assert "Early rather than middle or late" in bips.frame_rule_text(bips.DEFAULT_FRAME_FRACTION)
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    assert not any("frame-fraction" in r
                   for r in meta_of(out)["gate_disqualified_reasons"])


# ------------------------------------------------------------------------------------------------
# verdict
# ------------------------------------------------------------------------------------------------


def test_partially_filled_sheet_is_refused(corpus, tmp_path, stub_frames, capsys):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    fill(out, {e: "match" for e in eps[:-3]})          # last three left blank

    rc = run("verdict", "--sheet", str(out), "--styles", str(STYLES),
             "--out", str(tmp_path / "evidence.json"))
    assert rc == bips.EXIT_FATAL
    err = capsys.readouterr().err
    assert "blank `verdict`" in err
    for e in eps[-3:]:
        assert e in err
    assert not (tmp_path / "evidence.json").exists(), "a refusal must write nothing"


def test_illegal_verdict_token_is_refused(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    fill(out, {e: "match" for e in eps} | {eps[0]: "yes"})
    assert run("verdict", "--sheet", str(out), "--styles", str(STYLES),
               "--out", str(tmp_path / "evidence.json")) == bips.EXIT_FATAL


def test_mismatch_without_a_named_axis_is_refused(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    fill(out, {e: "match" for e in eps} | {eps[0]: "mismatch"})
    assert run("verdict", "--sheet", str(out), "--styles", str(STYLES),
               "--out", str(tmp_path / "evidence.json")) == bips.EXIT_FATAL


def test_changed_frame_bytes_are_fatal(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    fill(out, {e: "match" for e in eps})
    Path(rows_of(out)[0]["frame"]).write_bytes(b"a different picture")
    assert run("verdict", "--sheet", str(out), "--styles", str(STYLES),
               "--out", str(tmp_path / "evidence.json")) == bips.EXIT_FATAL


def test_verdict_emits_the_three_required_fields(corpus, tmp_path, stub_frames, capsys):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    verdicts = {e: "match" for e in eps}
    verdicts[eps[3]] = "mismatch"
    verdicts[eps[7]] = "unsure"
    fill(out, verdicts,
         axes={eps[3]: ["table", "lighting"]},
         notes={eps[3]: 'a light "grey" cloth, not black', eps[7]: "hand covers the apple"})

    art = tmp_path / "evidence.json"
    rc = run("verdict", "--sheet", str(out), "--styles", str(STYLES), "--out", str(art))
    ev = json.loads(art.read_text())

    # evidence_required: "the sampled episode ids, the sample size, and the per-episode verdicts"
    assert ev["sampled_episodes"] == eps
    assert ev["sample_size"] == 40
    assert ev["per_episode_verdicts"] == verdicts
    assert ev["verdict_counts"] == {"match": 38, "mismatch": 1, "unsure": 1}
    assert ev["mismatch_axis_counts"]["table"] == 1
    assert ev["disagreements"][0]["episode"] == eps[3]

    # ... and the pasteable form carries the same three, with the free text escaped so a quote in
    # somebody's note cannot break the committed partition it is pasted into.
    toml_text = capsys.readouterr().out
    assert "evidence_sample_size = 40" in toml_text
    assert f'evidence_sampled_episodes = [\n  "{eps[0]}"' in toml_text
    assert f'"{eps[3]} = mismatch"' in toml_text
    assert r'a light \"grey\" cloth' in toml_text
    assert "status" not in toml_text, "the verdict step must not pre-type the TODO's closure"

    # Inherited from the sheet: 120 episodes is not the committed 402.
    assert rc == bips.EXIT_NOT_GATE_QUALIFIED
    assert (art.parent / (art.name + ".sha256")).is_file()


def test_verdict_computes_no_overall_pass(corpus, tmp_path, stub_frames):
    """An inconstant appearance IS the finding, so no key in the artifact may adjudicate it."""
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    fill(out, {e: "mismatch" for e in eps},
         axes={e: ["apple"] for e in eps}, notes={e: "green apple" for e in eps})

    art = tmp_path / "evidence.json"
    run("verdict", "--sheet", str(out), "--styles", str(STYLES), "--out", str(art))
    ev = json.loads(art.read_text())

    assert not any(k in ev for k in ("pass", "passed", "overall", "verdict", "outcome", "ok"))
    # 40/40 mismatches is a fully answered sample. gate_qualified speaks to admissibility only, so
    # the only reason present is the one inherited from the synthetic corpus size.
    assert ev["coverage"] == 1.0
    assert all("402" in r for r in ev["gate_disqualified_reasons"])


def test_abstentions_below_the_coverage_floor_disqualify(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    eps = [r["episode"] for r in rows_of(out)]
    verdicts = {e: "match" for e in eps}
    for e in eps[:8]:                                   # 32/40 decidable = 0.80 < 0.90
        verdicts[e] = "unsure"
    fill(out, verdicts)

    art = tmp_path / "evidence.json"
    rc = run("verdict", "--sheet", str(out), "--styles", str(STYLES), "--out", str(art))
    ev = json.loads(art.read_text())
    assert rc == bips.EXIT_NOT_GATE_QUALIFIED
    assert ev["coverage"] == pytest.approx(0.80)
    assert any("coverage" in r for r in ev["gate_disqualified_reasons"])


def test_rows_from_another_sheet_are_refused(corpus, tmp_path, stub_frames):
    """Answers drawn under one sample must not be reported under another's provenance."""
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    rows = rows_of(out)
    for r in rows:
        r["verdict"] = "match"
    rows[0]["sheet_id"] = "0" * 64
    (out / "sheet.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    assert run("verdict", "--sheet", str(out), "--styles", str(STYLES),
               "--out", str(tmp_path / "evidence.json")) == bips.EXIT_FATAL


def test_a_reworded_committed_prompt_voids_the_sheet(corpus, tmp_path, stub_frames):
    """The verdicts answer a question about one string; if it changed, they answer nothing."""
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    fill(out, {r["episode"]: "match" for r in rows_of(out)})

    meta_path = out / "sheet_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["prompt"] = meta["prompt"].replace("red and yellow apple", "green apple")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    assert run("verdict", "--sheet", str(out), "--styles", str(STYLES),
               "--out", str(tmp_path / "evidence.json")) == bips.EXIT_FATAL


def test_missing_meta_is_refused(corpus, tmp_path, stub_frames):
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40")
    fill(out, {r["episode"]: "match" for r in rows_of(out)})
    (out / "sheet_meta.json").unlink()
    assert run("verdict", "--sheet", str(out), "--styles", str(STYLES),
               "--out", str(tmp_path / "evidence.json")) == bips.EXIT_FATAL


# ------------------------------------------------------------------------------------------------
# the sample's identity: deletion, duplication and substitution are all the same hole
# ------------------------------------------------------------------------------------------------


def test_deleted_rows_are_refused_not_reported_as_a_smaller_sample(corpus, tmp_path, stub_frames,
                                                                   capsys):
    """The partial-fill refusal only knows how to see a BLANK row. Deleting is easier and worse.

    Without the pin, five deleted rows gave exit 0 with gate_qualified true, sample_size 35,
    coverage 1.0 and a rule-of-three note recomputed to 3/35 — internally self-consistent evidence
    for a sample nobody drew, with five strata unrepresented and nothing in the artifact from which
    a reader could tell.
    """
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    dropped = [r["episode"] for r in rows[-5:]]
    write_rows(out, rows[:-5])

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    err = capsys.readouterr().err
    for e in dropped:
        assert e in err
    assert not art.exists(), "a refusal must write nothing"


def test_duplicated_rows_are_refused(corpus, tmp_path, stub_frames, capsys):
    """`sample_episode_ids` refuses a repeated episode at draw time; the verdict step must too.

    Admitted, one duplicated row reported sample_size 41 over 40 per-episode verdicts and coverage
    0.9756 instead of 1.0 — two wrong numbers, no crash.
    """
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    write_rows(out, rows + [dict(rows[0])])

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    assert "more than once" in capsys.readouterr().err
    assert not art.exists()


def test_a_substituted_episode_is_refused(corpus, tmp_path, stub_frames, capsys):
    """Same row count, same sheet_id, a different episode: the count would look untouched."""
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    swapped_in = "episode_000119"
    assert swapped_in not in {r["episode"] for r in rows}
    rows[0] = dict(rows[0], episode=swapped_in)
    write_rows(out, rows)

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    err = capsys.readouterr().err
    assert swapped_in in err and "never drawn" in err
    assert not art.exists()


def test_a_meta_edited_to_match_deleted_rows_no_longer_hashes_to_its_own_sheet_id(
        corpus, tmp_path, stub_frames, capsys):
    """Closing the loop: shrink the rows AND the meta's pinned list, and the sheet_id gives it up.

    sheet_id is a digest of seed, size, fraction, ids and prompt, and every row carries it, so a
    meta rewritten to agree with 35 surviving rows disagrees with the id all 35 of them carry.
    """
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    write_rows(out, rows[:-5])
    meta = meta_of(out)
    meta["sampled_episodes"] = [r["episode"] for r in rows[:-5]]
    meta["sample_size"] = 35
    meta["sample_strata"] = meta["sample_strata"][:-5]
    write_meta(out, meta)

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    assert "hash to" in capsys.readouterr().err
    assert not art.exists()


def test_the_evidence_carries_the_pinned_sample_so_a_reader_can_check_it(corpus, tmp_path,
                                                                        stub_frames):
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    art = tmp_path / "evidence.json"
    verdict_of(out, art)
    ev = json.loads(art.read_text())

    assert ev["sample_size_at_build"] == 40
    assert ev["sampled_episodes_at_build"] == [r["episode"] for r in rows]
    assert ev["sampled_episodes"] == ev["sampled_episodes_at_build"]
    assert len(ev["sample_strata"]) == 40


# ------------------------------------------------------------------------------------------------
# gate qualification is opt-in: it is read from the build's stamp, never inferred
# ------------------------------------------------------------------------------------------------


def test_a_meta_without_a_gate_stamp_is_refused_not_read_as_qualified(corpus, tmp_path,
                                                                      stub_frames, capsys):
    """`meta.get('gate_disqualified_reasons') or []` made TRUE the default for a missing key."""
    out = tmp_path / "sheet"
    build_and_fill(corpus, out)
    meta = meta_of(out)
    del meta["gate_disqualified_reasons"]
    write_meta(out, meta)

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    assert "opt-in" in capsys.readouterr().err
    assert not art.exists()


def test_a_meta_contradicting_its_own_gate_stamp_is_refused(corpus, tmp_path, stub_frames):
    """gate_qualified false with the reasons emptied used to produce gate_qualified TRUE evidence."""
    out = tmp_path / "sheet"
    build_and_fill(corpus, out)
    meta = meta_of(out)
    meta["gate_qualified"] = False
    meta["gate_disqualified_reasons"] = []
    write_meta(out, meta)

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    assert not art.exists()


@pytest.mark.parametrize("extra, hallmark", [
    (["--sample-size", "5"], "--sample-size 5"),
    (["--sample-size", "40", "--frame-fraction", "0.9"], "--frame-fraction 0.9"),
])
def test_a_hand_promoted_meta_cannot_launder_a_smoke_run(corpus, tmp_path, stub_frames,
                                                         extra, hallmark):
    """Emptying the reasons and setting gate_qualified true is self-consistent AND hashes right.

    Nothing in the sheet_id covers the reasons list, and a meta whose stamp and reasons agree
    passes every other check — so the two rules that are still derivable from the pinned sample are
    derived again here rather than merely inherited.
    """
    out = tmp_path / "sheet"
    build(corpus, out, *extra)
    fill(out, {r["episode"]: "match" for r in rows_of(out)})
    meta = meta_of(out)
    meta["gate_qualified"] = True
    meta["gate_disqualified_reasons"] = []
    write_meta(out, meta)

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_NOT_GATE_QUALIFIED
    ev = json.loads(art.read_text())
    assert ev["gate_qualified"] is False
    assert any(hallmark in r for r in ev["gate_disqualified_reasons"])


def test_an_inherited_reason_is_not_repeated(corpus, tmp_path, stub_frames):
    """The re-derivation above must produce the identical string, not a near-duplicate."""
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "5")
    fill(out, {r["episode"]: "match" for r in rows_of(out)})
    art = tmp_path / "evidence.json"
    verdict_of(out, art)
    reasons = json.loads(art.read_text())["gate_disqualified_reasons"]
    assert len(reasons) == len(set(reasons))
    assert sum("--sample-size 5" in r for r in reasons) == 1


def test_the_sheets_disqualification_reaches_the_evidence_and_the_toml(corpus, tmp_path,
                                                                      stub_frames, capsys):
    """The build's stamp is inherited rather than recomputed, and the pasteable fragment says so."""
    out = tmp_path / "sheet"
    build_and_fill(corpus, out)
    assert meta_of(out)["gate_qualified"] is False        # 120 episodes is not the committed 402

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_NOT_GATE_QUALIFIED
    ev = json.loads(art.read_text())
    assert ev["gate_qualified"] is False
    assert ev["gate_disqualified_reasons"]
    assert "evidence_gate_qualified = false" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------
# frame integrity: the digest is not optional
# ------------------------------------------------------------------------------------------------


def test_a_row_whose_digest_was_removed_is_refused(corpus, tmp_path, stub_frames, capsys):
    """The third case the two-branch check fell through: a frame named with no digest.

    Nulling one field was the cheapest way to exempt a row from the check the docstring calls
    fatal — here the bytes are changed too, and the old code returned gate_qualified true with no
    note anywhere.
    """
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    rows[0]["frame_sha256"] = None
    write_rows(out, rows)
    Path(rows[0]["frame"]).write_bytes(b"a different picture entirely")

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_FATAL
    assert "without its sha256" in capsys.readouterr().err
    assert not art.exists()


def test_a_row_whose_frame_path_was_removed_is_refused(corpus, tmp_path, stub_frames):
    """The mirror image: a digest with no frame is not a row build-sheet wrote either."""
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    rows[0]["frame"] = None
    write_rows(out, rows)
    assert verdict_of(out, tmp_path / "evidence.json") == bips.EXIT_FATAL


def test_rows_naming_no_pixels_at_all_cost_gate_qualification(corpus, tmp_path):
    """A --skip-frames sheet can be filled by a tool that never looked at anything. Stamped."""
    out = tmp_path / "sheet"
    build(corpus, out, "--sample-size", "40", "--skip-frames")
    fill(out, {r["episode"]: "match" for r in rows_of(out)})

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_NOT_GATE_QUALIFIED
    ev = json.loads(art.read_text())
    assert any("name no frame at all" in r for r in ev["gate_disqualified_reasons"])


def test_a_frame_deleted_after_the_fill_disqualifies_rather_than_passing(corpus, tmp_path,
                                                                        stub_frames):
    out = tmp_path / "sheet"
    rows = build_and_fill(corpus, out)
    Path(rows[0]["frame"]).unlink()

    art = tmp_path / "evidence.json"
    assert verdict_of(out, art) == bips.EXIT_NOT_GATE_QUALIFIED
    ev = json.loads(art.read_text())
    assert any("gone at verdict time" in r for r in ev["gate_disqualified_reasons"])


# ------------------------------------------------------------------------------------------------
# a hand-edited JSONL is the intended workflow, so malformed JSON is an expected input
# ------------------------------------------------------------------------------------------------


def test_a_broken_sheet_line_exits_fatal_naming_the_line(corpus, tmp_path, stub_frames, capsys):
    out = tmp_path / "sheet"
    build_and_fill(corpus, out)
    text = (out / "sheet.jsonl").read_text().splitlines()
    text[3] = text[3][:-12]                      # a truncated line, the way an editor leaves one
    (out / "sheet.jsonl").write_text("\n".join(text) + "\n")

    assert verdict_of(out, tmp_path / "evidence.json") == bips.EXIT_FATAL
    err = capsys.readouterr().err
    assert "FATAL:" in err and "sheet.jsonl:4" in err, "a traceback names neither file nor row"
    assert "Traceback" not in err


def test_a_broken_meta_exits_fatal(corpus, tmp_path, stub_frames, capsys):
    out = tmp_path / "sheet"
    build_and_fill(corpus, out)
    (out / "sheet_meta.json").write_text('{"schema": "wam.identity_prompt_sheet/1",')

    assert verdict_of(out, tmp_path / "evidence.json") == bips.EXIT_FATAL
    assert "not valid JSON" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------
# the committed partition this is evidence about
# ------------------------------------------------------------------------------------------------


def test_reads_the_committed_identity_prompt_and_its_provenance():
    identity = bips.read_identity_style(STYLES)
    assert identity["style_id"] == "identity-source"
    assert identity["repeats"] == 10
    assert "episode_000135_clip000" in identity["caption_provenance"]
    assert identity["todo_status"] == "OPEN", (
        "this harness exists to produce evidence for an OPEN TODO; if it has closed, the "
        "evidence it closed with is what should be read, not regenerated"
    )
    assert len(identity["partition_content_sha256"]) == 64
