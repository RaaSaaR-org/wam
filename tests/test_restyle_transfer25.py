"""The restyle driver's refusals, the seed channel arm C depends on, and PR-08 §6's G0c composite.

These run entirely on the null backend: no GPU, no Transfer2.5 checkout, no weights. What they
exercise is everything AROUND the model call, which is where PR-08's controls actually live — the
seed reaching the sampler, the per-unit isolation, the status file being written after the mp4 is
asserted, the four things the driver refuses to default, and the one gate that is solved by
construction and therefore has to be in the generation path rather than in a checker.

THE ROBOT MASKER IS FAKED HERE AND THE COMPOSITE IS NOT. ``robot_composite.build_masker`` is
monkeypatched to a masker whose masks are known rectangles — not because the real one cannot run
(the pinned GroundingDINO and SAM 2 checkpoints are staged), but because a test whose expected
pixels depend on what a segmenter finds is a test of the segmenter. Everything downstream of the
mask — the compositing arithmetic, both refusals, the IoU, the cache, the record, the quarantine and
the driver's wiring — is the real code. A monkeypatched factory is test instrumentation and not a
code path: the tests at the bottom of this file are the ones that assert no code path exists.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import restyle_transfer25 as rt  # noqa: E402
import robot_composite as rc  # noqa: E402

STYLE_SET = "train"

#: Small enough that a whole test suite of h264 round trips is free, large enough that a mask
#: rectangle, its complement and an IoU over both are all non-degenerate.
FRAME_H, FRAME_W = 48, 64


def _write(path: pathlib.Path, payload) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_clip(path: pathlib.Path, frames: int, *, tint: int = 0) -> pathlib.Path:
    """A real, decodable mp4. The fixture's sources have to be real now that G0c decodes them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((frames, FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for i in range(frames):
        data[i, :, :, 0] = (i * 3 + tint) % 256
        data[i, :, :, 1] = (i * 5 + tint) % 256
        data[i, :, :, 2] = (i * 7 + tint) % 256
    rc.encode_clip(data, path, 30.0)
    return path


class FakeMasker:
    """A robot mask that is a known rectangle, so every pixel of the composite is predictable.

    It answers differently for source and generated frames — the generated frames the null backend
    writes are flat colour fields, and the IoU diagnostic is only meaningful if the two masks can
    disagree. ``rows``/``cols`` are per-instance so a test can make the mask empty or the whole
    frame without reaching into the compositor.
    """

    def __init__(self, rows=(4, 20), cols=(4, 20), gen_shift: int = 2) -> None:
        self.rows, self.cols, self.gen_shift = rows, cols, gen_shift
        self.calls = 0
        self.preflighted = 0

    def preflight(self) -> None:
        self.preflighted += 1

    def provenance(self) -> dict:
        """Every field :func:`rc.segmenter_identity` reads, because the loader compares them all.

        A fake that declared fewer of them would make the committed bound's cross-check pass by
        absence on both sides, which is the absence-permissive comparison the loader exists to
        refuse.
        """
        return {
            "name": "fake-robot-masker",
            "version": "test",
            "prompt": rc.ROBOT_TEXT_PROMPT,
            "box_threshold": 0.15,
            "text_threshold": 0.25,
            "box_rule": "test rectangle",
            "upstream_retry_not_run": "test masker; there is no upstream retry to skip",
        }

    def mask(self, rgb) -> np.ndarray:
        self.calls += 1
        arr = np.asarray(rgb)
        out = np.zeros(arr.shape[:2], dtype=bool)
        # The null backend's frames are constant across a frame; the fixture's sources are too, but
        # with a different per-channel recipe. Telling them apart by variance would be fragile, so
        # the shift is applied to every frame and the source/generated distinction is made by the
        # caller passing raw generated frames only into the IoU sample.
        r0, r1 = self.rows
        c0, c1 = self.cols
        out[r0:r1, c0:c1] = True
        return out


class ShiftedMasker(FakeMasker):
    """The same rectangle, moved once the source pass is over.

    The IoU diagnostic needs two masks that CAN disagree, and ``composite_clip`` takes all the
    source masks first (one pass over the source clip) and only then samples the generated frames.
    So the first ``source_frames`` calls answer for the source and everything after answers for the
    generator. Counting calls rather than inspecting pixels: the fixture's frames and the null
    backend's are both flat fields, and a masker that guessed from content would be testing itself.
    """

    def __init__(self, source_frames: int, **kw) -> None:
        super().__init__(**kw)
        self.source_frames = source_frames

    def mask(self, rgb) -> np.ndarray:
        base = super().mask(rgb)
        if self.calls <= self.source_frames:
            return base
        shifted = np.zeros_like(base)
        shifted[
            self.rows[0] + self.gen_shift : self.rows[1] + self.gen_shift,
            self.cols[0] : self.cols[1],
        ] = True
        return shifted


#: The sha256 the bound fixture claims to have been measured over before any corpus exists. The
#: ``corpus`` fixture overwrites it with the real manifest's, because the loader now refuses a bound
#: measured over a different corpus and a fixture that could not pass that check would only be
#: testing the fixture.
UNPINNED_MANIFEST_SHA = "0" * 64


@pytest.fixture(autouse=True)
def bound(tmp_path: pathlib.Path) -> pathlib.Path:
    """A committed-shaped area bound. Every field the loader requires, because it requires them.

    ``estimator`` is the fake masker's own provenance rather than a hand-written stub: the loader
    compares the bound's segmenter to the one that will make the masks, so a stub here would make
    every composite test fail on a disagreement that is an artefact of the fixture.
    """
    return _write(
        tmp_path / "pr08_robot_mask_area.json",
        {
            "max_frame_fraction": 0.5,
            "bound_rationale": "test fixture; the real one is a decision with a written argument.",
            "measured": {"frames": 10, "max": 0.2, "limit": None, "stride": 1},
            "measurement_qualified": True,
            "estimator": FakeMasker().provenance(),
            "prompt": rc.ROBOT_TEXT_PROMPT,
            "source_manifest_sha256": UNPINNED_MANIFEST_SHA,
        },
    )


@pytest.fixture(autouse=True)
def masker(monkeypatch):
    """Every test in this file gets the fake masker; none of them can reach the real one."""
    instance = FakeMasker()
    monkeypatch.setattr(rc, "build_masker", lambda: instance)
    return instance


@pytest.fixture()
def corpus(tmp_path: pathlib.Path, bound: pathlib.Path):
    """A two-episode source, a two-style partition, and a work list over them.

    It also PINS the area-bound fixture to this manifest. A bound is a claim about a corpus and the
    loader refuses one whose ``source_manifest_sha256`` belongs to another, so the fixture that
    supplies the corpus is the only thing that can honestly fill that field in.
    """
    src = tmp_path / "source"
    _write_clip(src / "videos" / "ep000.mp4", 12, tint=0)
    _write_clip(src / "videos" / "ep001.mp4", 9, tint=11)
    manifest = _write(
        src / "manifest.json",
        {
            "resolution": [640, 480],
            "episodes": [
                {"id": "ep000", "frames": 12, "video": "videos/ep000.mp4"},
                {"id": "ep001", "frames": 9, "video": "videos/ep001.mp4"},
            ],
        },
    )
    styles = _write(
        tmp_path / "styles.json",
        {
            STYLE_SET: [
                {"id": "train-01", "prompt": "warm afternoon light through a window"},
                {"id": "train-02", "prompt": "cool overcast daylight"},
            ],
            "eval": [{"id": "eval-01", "prompt": "harsh light from one side"}],
        },
    )
    rows = [
        {"unit": "ep000__train-01__r00", "episode": "ep000", "frames": 12,
         "style": "train-01", "repeat": 0, "seed": 7001},
        {"unit": "ep001__train-02__r00", "episode": "ep001", "frames": 9,
         "style": "train-02", "repeat": 0, "seed": 7002},
    ]
    work = tmp_path / "work.jsonl"
    work.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    payload = json.loads(bound.read_text())
    payload["source_manifest_sha256"] = rc._file_sha256(manifest)
    _write(bound, payload)
    return {
        "manifest": manifest,
        "styles": styles,
        "work": work,
        "out": tmp_path / "raw",
        "rows": rows,
        "root": tmp_path,
    }


def _argv(corpus, over: dict | None = None) -> list[str]:
    args = {
        "--checkpoint-path": "/staged/ckpt",
        "--manifest": str(corpus["manifest"]),
        "--styles": str(corpus["styles"]),
        "--style-set": STYLE_SET,
        "--work-list": str(corpus["work"]),
        "--out": str(corpus["out"]),
        "--control": "depth:0.5",
        "--backend": "null",
        "--robot-mask-area-bound": str(corpus["root"] / "pr08_robot_mask_area.json"),
    }
    args.update(over or {})
    # A None value is a store_true flag, which takes no argument.
    flat = [x for k, v in args.items() for x in ((k,) if v is None else (k, v))]
    return flat + ["--no-guardrails"]


def _context(bound_path: pathlib.Path, corpus, **kw) -> rc.CompositeContext:
    return rc.build_context(
        area_bound_path=bound_path, source_manifest=corpus["manifest"], **kw
    )


def _cross_checked(bound_path: pathlib.Path, corpus, masker=None) -> rc.AreaBound:
    """The bound as a compositing path gets it: checked against this segmenter and this corpus.

    Tests that build a :class:`rc.CompositeContext` by hand go through here rather than calling
    ``load_area_bound`` bare, because a bare load returns ``cross_checked=False`` and the context
    refuses it — which is the point of the refusal, and would otherwise show up as every composite
    test failing for the same uninformative reason.
    """
    return rc.load_area_bound(
        bound_path,
        expect_segmenter=(masker or FakeMasker()).provenance(),
        expect_source_manifest=corpus["manifest"],
    )


# -- the happy path, and the layout 97 requires ------------------------------------------------


def test_a_clean_run_writes_the_per_unit_layout_the_harvest_reads(corpus, capsys):
    assert rt.main(_argv(corpus)) == 0
    for row in corpus["rows"]:
        unit = corpus["out"] / row["unit"]
        assert (unit / "vision.mp4").is_file()
        record = json.loads((unit / "sample_outputs.json").read_text())
        assert record["status"] == "success"
        # The seed is in the record, which is what makes one clip reproducible from the record
        # alone — 97's stated reason for carrying it in the work unit at all.
        assert record["seed"] == row["seed"]
        assert record["backend"] == "null"


def test_the_status_file_is_written_after_the_video_not_before(corpus, monkeypatch):
    """Upstream's own sidecar lands BEFORE generation, so presence never means success."""
    def backend_that_writes_nothing(sample, out_dir):
        return {"backend": "null"}

    monkeypatch.setattr(rt, "_null_backend", backend_that_writes_nothing)
    assert rt.main(_argv(corpus)) == 0
    unit = corpus["out"] / corpus["rows"][0]["unit"]
    assert not (unit / "vision.mp4").exists()
    record = json.loads((unit / "sample_outputs.json").read_text())
    assert record["status"] == "error"
    assert "missing or empty" in record["detail"]


def test_a_stale_success_is_withdrawn_before_the_unit_is_retried(corpus, monkeypatch):
    unit_dir = corpus["out"] / corpus["rows"][0]["unit"]
    unit_dir.mkdir(parents=True)
    (unit_dir / "sample_outputs.json").write_text('{"status": "success"}', encoding="utf-8")

    def exploding(sample, out_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(rt, "_null_backend", exploding)
    assert rt.main(_argv(corpus)) == 0
    assert json.loads((unit_dir / "sample_outputs.json").read_text())["status"] == "error"


# -- the seed channel arm C depends on ----------------------------------------------------------


def test_the_same_seed_reproduces_the_clip_and_a_different_seed_does_not(corpus, tmp_path):
    """Arm C is ten samples of ONE prompt. If the seed does not reach the sampler the control has
    measured nothing, so the null backend makes the seed observable in the bytes."""
    base = rt.build_sample(
        rt.WorkUnit("u", "ep000", 12, "train-01", 0, 7001),
        source_root=corpus["manifest"].parent,
        episode={"id": "ep000", "frames": 12, "video": "videos/ep000.mp4"},
        style={"id": "train-01", "prompt": "p"},
        controls=[rt.Control("depth", 0.5)],
        bucket="480",
    )
    other = dict(base, seed=7002)
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    for d in (a, b, c):
        d.mkdir()
    rt._null_backend(base, a)
    rt._null_backend(base, b)
    rt._null_backend(other, c)
    assert (a / "vision.mp4").read_bytes() == (b / "vision.mp4").read_bytes()
    assert (a / "vision.mp4").read_bytes() != (c / "vision.mp4").read_bytes()


def test_a_row_without_a_seed_is_refused_by_name(corpus):
    corpus["work"].write_text(
        json.dumps({"unit": "u", "episode": "ep000", "frames": 12, "style": "train-01", "repeat": 0})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rt.DriverError, match="seed"):
        rt.load_work_list(corpus["work"])


def test_a_non_integer_seed_is_refused(corpus):
    corpus["work"].write_text(
        json.dumps({"unit": "u", "episode": "ep000", "frames": 12, "style": "train-01",
                    "repeat": 0, "seed": "7001"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rt.DriverError, match="not an int"):
        rt.load_work_list(corpus["work"])


def test_a_duplicate_unit_is_refused(corpus):
    row = dict(corpus["rows"][0])
    corpus["work"].write_text("\n".join([json.dumps(row), json.dumps(row)]) + "\n", encoding="utf-8")
    with pytest.raises(rt.DriverError, match="repeats unit"):
        rt.load_work_list(corpus["work"])


# -- the four refusals --------------------------------------------------------------------------


def test_guardrails_may_not_be_left_on(corpus):
    argv = [a for a in _argv(corpus) if a != "--no-guardrails"]
    assert rt.main(argv) == 1


def test_the_control_block_has_no_default(corpus):
    with pytest.raises(rt.DriverError, match="uncommitted"):
        rt.parse_controls("")


@pytest.mark.parametrize("spec", ["sharpen:0.5", "depth", "depth:1.5", "depth:x"])
def test_a_malformed_control_spec_is_refused(spec):
    with pytest.raises(rt.DriverError):
        rt.parse_controls(spec)


def test_the_120x160_corpus_is_refused_by_resolution(tmp_path):
    man = _write(tmp_path / "manifest.json",
                 {"resolution": [120, 160], "episodes": [{"id": "e", "frames": 1, "video": "v.mp4"}]})
    with pytest.raises(rt.DriverError, match="640x480|PR-08"):
        rt.load_manifest(man)


def test_only_640x480_maps_to_a_bucket():
    assert rt.resolve_resolution("640x480") == ("480", "4,3")
    with pytest.raises(rt.DriverError, match="bucket"):
        rt.resolve_resolution("1280x720")


# -- per-unit isolation, which upstream does not provide ----------------------------------------


def test_one_failing_unit_does_not_take_the_others_with_it(corpus, monkeypatch):
    """Upstream's generate() has no try/except and keep_going covers only guardrail blocks, so on
    a 10 050-clip run this is the difference between losing a clip and losing a chunk."""
    real = rt._null_backend

    def flaky(sample, out_dir):
        if sample["name"].startswith("ep000"):
            raise RuntimeError("unreadable video")
        return real(sample, out_dir)

    monkeypatch.setattr(rt, "_null_backend", flaky)
    assert rt.main(_argv(corpus)) == 0
    first, second = (corpus["out"] / r["unit"] for r in corpus["rows"])
    assert json.loads((first / "sample_outputs.json").read_text())["status"] == "error"
    assert json.loads((second / "sample_outputs.json").read_text())["status"] == "success"
    assert (second / "vision.mp4").is_file()


def test_a_work_list_that_disagrees_with_the_manifest_is_fatal_not_per_unit(corpus):
    corpus["work"].write_text(
        json.dumps({"unit": "u", "episode": "ep999", "frames": 1, "style": "train-01",
                    "repeat": 0, "seed": 1}) + "\n",
        encoding="utf-8",
    )
    assert rt.main(_argv(corpus)) == 1


def test_an_invented_style_is_refused_rather_than_synthesised(corpus):
    with pytest.raises(rt.DriverError, match="COMMITTED"):
        rt.load_styles(corpus["styles"], "identity")


# -- sharding, and the N-fold overspend it exists to prevent -------------------------------------


def test_nproc_greater_than_one_refuses_to_run_unsharded(corpus, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(rt.DriverError, match="torchrun|RANK"):
        rt.shard(rt.load_work_list(corpus["work"]), 4)


def test_each_rank_takes_a_disjoint_share(corpus, monkeypatch):
    units = rt.load_work_list(corpus["work"])
    seen = []
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        seen.append({u.unit for u in rt.shard(units, 2)})
    assert seen[0].isdisjoint(seen[1])
    assert seen[0] | seen[1] == {u.unit for u in units}


# -- the manifest map, and the estimator swap it prevents ---------------------------------------


def test_a_manifest_supplied_depth_map_is_passed_rather_than_re_estimated(corpus):
    src = corpus["manifest"].parent
    (src / "depth").mkdir()
    (src / "depth" / "ep000.mp4").write_bytes(b"depth")
    sample = rt.build_sample(
        rt.WorkUnit("u", "ep000", 12, "train-01", 0, 7001),
        source_root=src,
        episode={"id": "ep000", "frames": 12, "video": "videos/ep000.mp4", "depth": "depth/ep000.mp4"},
        style={"id": "train-01", "prompt": "p"},
        controls=[rt.Control("depth", 0.5)],
        bucket="480",
    )
    assert sample["depth"]["control_path"].endswith("depth/ep000.mp4")


def test_a_declared_but_missing_map_is_refused_rather_than_silently_re_estimated(corpus):
    with pytest.raises(rt.DriverError, match="Refusing"):
        rt.build_sample(
            rt.WorkUnit("u", "ep000", 12, "train-01", 0, 7001),
            source_root=corpus["manifest"].parent,
            episode={"id": "ep000", "frames": 12, "video": "videos/ep000.mp4", "depth": "gone.mp4"},
            style={"id": "train-01", "prompt": "p"},
            controls=[rt.Control("depth", 0.5)],
            bucket="480",
        )


# -- what job 189142 cost us -------------------------------------------------------------------
#
# The restyle job on 2026-08-20 exited 0:0, wrote THROUGHPUT.json, and reported 9.56 GPU-h per
# variant to the PR-08 §8 item 3 budget. It had generated nothing: every unit died inside
# SetupArguments and the wall clock the sbatch measured was the time to reach the crash. The three
# tests below are the three links in that chain, each one broken on its own.


def test_a_dead_unit_cannot_become_a_measurement(corpus, monkeypatch):
    """--require-success is what the TIMING path passes. Without it the driver returns 0 on a
    chunk of corpses (deliberately — the chunked run resumes), and the sbatch above it times the
    corpse and calls the number throughput."""

    def dead(sample, out_dir):
        raise RuntimeError("SetupArguments: model is required")

    monkeypatch.setattr(rt, "_null_backend", dead)
    assert rt.main(_argv(corpus)) == 0, "the resumable default must not change"
    assert rt.main(_argv(corpus, {"--require-success": None})) == 1


def test_require_success_is_silent_when_every_unit_lands(corpus):
    """It must gate on the outcome, not merely on being passed."""
    assert rt.main(_argv(corpus, {"--require-success": None})) == 0


def test_the_setup_dict_names_a_model_because_upstream_validates_before_defaults(corpus, monkeypatch):
    """SetupArguments declares `model` with a default, and that default is unreachable: its
    validator is mode="before", so it sees the raw dict and raises "model is required" for a key
    pydantic would have filled in a moment later. The stub reproduces that ordering exactly."""
    seen = {}

    class _SetupArguments:
        def __init__(self, **kw):
            if kw.get("model") is None:  # config.py:263-270, mode="before"
                raise ValueError("model is required")
            seen.update(kw)
            self.model_key = type("K", (), {"distilled": False})()

    class _InferenceArguments:
        def __init__(self, **kw):
            self.name = kw["name"]
            self.video_path = kw["video_path"]

    class _Control2WorldInference:
        def __init__(self, args, hint_keys):
            seen["hint_keys"] = list(hint_keys)
            self.checkpoint_list = [f"s3://stub/{k}" for k in hint_keys]

        def generate(self, samples, out_dir):
            # A decodable clip at the source's geometry, because G0c composites the framework's
            # output like anything else and a stub that wrote `b"stub mp4"` would be testing the
            # composite's decode refusal instead of the checkpoint bookkeeping this test is about.
            path = out_dir / f"{samples[0].name}.mp4"
            source = rc.decode_clip(pathlib.Path(samples[0].video_path))
            rc.encode_clip(np.zeros_like(source), path, 30.0)
            return [path]

    monkeypatch.setitem(
        sys.modules,
        "cosmos_transfer2.config",
        type(sys)("cosmos_transfer2.config"),
    )
    monkeypatch.setitem(
        sys.modules,
        "cosmos_transfer2.inference",
        type(sys)("cosmos_transfer2.inference"),
    )
    sys.modules["cosmos_transfer2.config"].SetupArguments = _SetupArguments
    sys.modules["cosmos_transfer2.config"].InferenceArguments = _InferenceArguments
    sys.modules["cosmos_transfer2.inference"].Control2WorldInference = _Control2WorldInference

    argv = _argv(corpus, {"--backend": "transfer25", "--control": "depth:0.5,seg:0.5"})
    assert rt.main(argv) == 0
    assert seen["model"] == "depth", "with several controls upstream ignores it, but it must be one of them"
    assert seen["hint_keys"] == ["depth", "seg"]

    record = json.loads((corpus["out"] / corpus["rows"][0]["unit"] / "sample_outputs.json").read_text())
    # Two hint keys means upstream resolved its own checkpoints and never looked at
    # --checkpoint-path. The record has to say so, because the sbatch's log line does not.
    assert record["checkpoint_path_honoured"] is False
    assert record["checkpoints_loaded"] == ["s3://stub/depth", "s3://stub/seg"]


# ================================================================================================
# PR-08 §6 G0c — the real robot's pixels, composited back over every generated frame
# ================================================================================================
#
# G0c is the one gate in §6 that is solved BY CONSTRUCTION, which is why these tests are here and
# not in a test file for a checker. Its sentence is "the defect cannot enter", not "the defect is
# detected": video_fidelity has been measured against the generic-manipulator defect and cannot see
# it (runs/backbone_eval/video/embodiment_grid.png), and §6 refuses an IoU threshold on the robot
# mask in the same breath. A downstream checker would therefore be a checker with no detector.
#
# So what has to be true is a property of the DRIVER, not of an artifact, and the last group below
# is the one that matters most: there must be no way to reach a filed clip that skipped this.


# -- the arithmetic: exact source inside, exact generated outside ---------------------------------


def test_the_composite_is_the_source_inside_the_mask_and_the_generated_pixels_outside_it():
    """The whole of G0c in one assertion, at the only place it can be checked exactly.

    Pixel exactness lives in the ARRAY and not in the mp4: the composited clip is re-encoded with
    h264, which is lossy, exactly as the generator's own output already was. The claim G0c makes —
    "the defect cannot enter" — is a claim about what this function computes, and it is only true if
    the robot region carries no generated information at all.
    """
    source = np.full((8, 12, 3), 200, dtype=np.uint8)
    generated = np.full((8, 12, 3), 40, dtype=np.uint8)
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:5, 3:9] = True

    out = rc.composite_frame(source, generated, mask)

    assert np.array_equal(out[mask], source[mask]), "inside the mask it must be the SOURCE, exactly"
    assert np.array_equal(out[~mask], generated[~mask]), "outside it, the GENERATED pixels, exactly"


def test_the_mask_edge_is_hard_so_no_generated_pixel_is_blended_onto_the_robot():
    """A feather would put generated pixels on the robot's SILHOUETTE, which is where the
    generic-manipulator defect lives. Every output pixel must be byte-identical to one input or the
    other; a blend would produce a third value, and no value between 40 and 200 may appear."""
    source = np.full((16, 16, 3), 200, dtype=np.uint8)
    generated = np.full((16, 16, 3), 40, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True

    out = rc.composite_frame(source, generated, mask)

    assert set(np.unique(out).tolist()) == {40, 200}, "a third value means a blended edge"


def test_a_mask_that_does_not_fit_the_frame_is_refused_rather_than_broadcast():
    with pytest.raises(rc.CompositeError, match="does not fit"):
        rc.composite_frame(
            np.zeros((8, 12, 3), np.uint8), np.zeros((8, 12, 3), np.uint8), np.zeros((4, 6), bool)
        )


# -- the two refusals ----------------------------------------------------------------------------


def test_an_empty_robot_mask_refuses_the_clip_rather_than_compositing_nothing(corpus, bound, tmp_path):
    """An empty mask makes the composite the identity, which means the GENERATED manipulator went
    straight into the corpus. That is the exact failure G0c exists to make impossible, so it is a
    refusal and not a warning, and there is no number in the check to loosen."""
    empty = FakeMasker(rows=(0, 0), cols=(0, 0))
    context = rc.CompositeContext(
        masker=empty, bound=_cross_checked(bound, corpus, empty), iou_stride=10, cache=None
    )
    generated = _write_clip(tmp_path / "gen_empty.mp4", 12)

    with pytest.raises(rc.CompositeError, match="EMPTY"):
        context.composite(
            source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",
            generated_video=generated,
        )


def test_an_implausibly_large_robot_mask_refuses_the_clip(corpus, bound, tmp_path):
    """A mask that has grounded on the table composites the SOURCE back over everything: the
    restyle becomes a no-op and arms B and C silently become arm A, at full GPU cost."""
    everything = FakeMasker(rows=(0, FRAME_H), cols=(0, FRAME_W))
    context = rc.CompositeContext(
        masker=everything, bound=_cross_checked(bound, corpus, everything), iou_stride=10, cache=None
    )
    generated = _write_clip(tmp_path / "gen_big.mp4", 12)

    with pytest.raises(rc.CompositeError, match="above the committed bound"):
        context.composite(
            source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",
            generated_video=generated,
        )


def test_one_bad_frame_out_of_many_still_refuses_the_whole_clip(corpus, bound, tmp_path):
    """"9 frames out of 427 had no robot composited" is not a smaller version of the failure: those
    9 carry a generated manipulator into the training set exactly as 427 would, and nothing
    downstream reads a corpus frame by frame."""

    class OneEmptyFrame(FakeMasker):
        def mask(self, rgb):
            out = super().mask(rgb)
            return np.zeros_like(out) if self.calls == 3 else out

    context = rc.CompositeContext(
        masker=OneEmptyFrame(), bound=_cross_checked(bound, corpus), iou_stride=10, cache=None
    )
    generated = _write_clip(tmp_path / "gen_one_bad.mp4", 12)

    with pytest.raises(rc.CompositeError, match="frame 2"):
        context.composite(
            source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",
            generated_video=generated,
        )


def test_clips_that_cannot_be_paired_frame_for_frame_are_refused(corpus, bound, tmp_path):
    """Compositing generated frame i over source frame j puts the robot from one instant into the
    scene of another — geometry drift manufactured by the gate that protects geometry, which G0b
    would then score as a generator defect."""
    context = _context(bound, corpus, cache_dir=None)
    short = _write_clip(tmp_path / "gen_short.mp4", 5)

    with pytest.raises(rc.CompositeError, match="same clip at the same"):
        context.composite(
            source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",  # 12 frames
            generated_video=short,
        )


def test_a_source_whose_length_disagrees_with_the_manifest_is_refused(corpus, bound, tmp_path):
    """The actions are carried over by INDEX, so a source longer or shorter than the label column
    pairs every frame after the divergence with the wrong action — silently, no decode error."""
    context = _context(bound, corpus, cache_dir=None)
    generated = _write_clip(tmp_path / "gen12.mp4", 12)

    with pytest.raises(rc.CompositeError, match="the manifest declares"):
        context.composite(
            source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",  # really 12
            generated_video=generated,
            expected_frames=999,
        )


# -- the area bound: measured, or refused, never coined -------------------------------------------


def test_a_missing_area_bound_refuses_and_names_the_measurement_rather_than_defaulting(tmp_path):
    """PR-08 §6 opens "No threshold is coined." There is no default here and there must not be: a
    silent 0.6 would be a threshold pre-registered by a library default."""
    with pytest.raises(rc.CompositeError) as excinfo:
        rc.load_area_bound(tmp_path / "not_there.json")
    message = str(excinfo.value)
    assert "robot_composite.py measure" in message, "the refusal has to name what to measure"
    assert "coined" in message


def test_a_bound_with_no_measurement_behind_it_is_refused(tmp_path):
    """A file saying only {"max_frame_fraction": 0.6} is a coined number in a committed file's
    clothing: it cannot say which segmenter measured the distribution the bound sits above."""
    path = _write(tmp_path / "bare.json", {"max_frame_fraction": 0.6})
    with pytest.raises(rc.CompositeError, match="missing"):
        rc.load_area_bound(path)


def test_a_measured_distribution_with_no_decided_bound_is_refused(tmp_path, bound):
    """This is what the measure mode writes. The bound is a decision with a written rationale, and
    the null is there so that nobody mistakes "measured" for "decided"."""
    payload = json.loads(bound.read_text())
    payload["max_frame_fraction"] = None
    path = _write(tmp_path / "null_bound.json", payload)
    with pytest.raises(rc.CompositeError, match="has been measured and the bound has not"):
        rc.load_area_bound(path)


def test_a_bound_with_no_written_rationale_is_refused(tmp_path, bound):
    payload = json.loads(bound.read_text())
    payload["bound_rationale"] = "   "
    path = _write(tmp_path / "no_why.json", payload)
    with pytest.raises(rc.CompositeError, match="rationale"):
        rc.load_area_bound(path)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5, "0.5", True])
def test_a_bound_that_is_not_a_fraction_of_a_frame_is_refused(tmp_path, bound, value):
    payload = json.loads(bound.read_text())
    payload["max_frame_fraction"] = value
    path = _write(tmp_path / "odd.json", payload)
    with pytest.raises(rc.CompositeError):
        rc.load_area_bound(path)


def test_a_bound_of_exactly_one_is_refused_because_it_would_delete_the_over_large_check(
    tmp_path, bound
):
    """1.0 is not a loose bound, it is the over-large refusal switched off in a committed file.

    ``check_mask`` refuses on ``fraction > bound.max_frame_fraction`` and an area fraction cannot
    exceed 1.0, so at exactly 1.0 that branch is unreachable. Every other validation still passes —
    the shape is right, the rationale is there — and the failure is silent, because the empty-mask
    half keeps firing: over-large masks composite the source back over the whole frame, the restyle
    becomes a no-op, and arms B and C become arm A at full GPU cost.
    """
    payload = json.loads(bound.read_text())
    payload["max_frame_fraction"] = 1.0
    path = _write(tmp_path / "one.json", payload)

    with pytest.raises(rc.CompositeError, match="outside .0, 1."):
        rc.load_area_bound(path)


def test_a_bound_measured_by_a_different_segmenter_is_refused(tmp_path, bound, corpus):
    """Re-pinning the detector invalidates every cached mask; it must invalidate the bound too.

    That is the whole rule, and it is why the cross-check reads the same field tuple the cache key
    does. A bound measured under other weights sits above a different area distribution: it either
    never fires — and over-large masks composite the source back over the whole frame — or fires on
    everything. Two copies of one measurement drifting apart is the failure this project has already
    paid for once.
    """
    stale = json.loads(bound.read_text())
    stale["estimator"] = dict(FakeMasker().provenance(), version="grounding-dino@an-older-pin")
    path = _write(tmp_path / "stale_segmenter.json", stale)

    with pytest.raises(rc.CompositeError, match="DIFFERENT segmenter"):
        _cross_checked(path, corpus)


def test_a_bound_measured_over_a_different_source_corpus_is_refused(tmp_path, bound, corpus):
    """A bound is a claim about a corpus: the largest fraction of a frame a robot covers in THESE
    episodes. Held over other episodes it is a number with no distribution behind it, which is the
    coined threshold PR-08 §6 opens by refusing."""
    elsewhere = json.loads(bound.read_text())
    elsewhere["source_manifest_sha256"] = UNPINNED_MANIFEST_SHA
    path = _write(tmp_path / "other_corpus.json", elsewhere)

    with pytest.raises(rc.CompositeError, match="DIFFERENT source corpus"):
        _cross_checked(path, corpus)


def test_a_bound_measured_under_a_different_prompt_is_refused(tmp_path, bound):
    """No caller input is needed for this one — the prompt is a committed constant in this build, so
    a bound measured under another one is stale on its face. A narrower prompt yields a smaller mask
    and a smaller distribution, and a bound above THAT one is not above this one."""
    payload = json.loads(bound.read_text())
    payload["prompt"] = "robot."
    path = _write(tmp_path / "other_prompt.json", payload)

    with pytest.raises(rc.CompositeError, match="ROBOT_TEXT_PROMPT"):
        rc.load_area_bound(path)


def test_a_bound_sitting_above_a_truncated_measurement_is_refused(tmp_path, bound):
    """A distribution over three episodes at stride 30 must not be indistinguishable, at load time,
    from one over the whole corpus. measure_geom_tol refuses exactly this for exactly this reason,
    and the loader has to read the stamp or the stamp is decoration."""
    smoke = json.loads(bound.read_text())
    smoke["measurement_qualified"] = False
    smoke["measurement_disqualified_reasons"] = ["--limit 3: 3 of the manifest's 402 episodes"]
    path = _write(tmp_path / "smoke.json", smoke)

    with pytest.raises(rc.CompositeError) as excinfo:
        rc.load_area_bound(path)
    assert "smoke run" in str(excinfo.value)
    assert "402 episodes" in str(excinfo.value), "the artifact's own reason has to reach the reader"


def test_a_bound_that_was_never_cross_checked_cannot_reach_a_composite(bound, corpus):
    """Validated and accepted-for-this-run are two different things, and only the second may
    composite. The refusal lives in CompositeContext rather than in build_context because this is
    the type every compositing path goes through — including one built by hand."""
    validated = rc.load_area_bound(bound)
    assert validated.cross_checked is False

    with pytest.raises(rc.CompositeError, match="never cross-checked"):
        rc.CompositeContext(masker=FakeMasker(), bound=validated, iou_stride=10, cache=None)


def test_the_cache_key_and_the_bound_cross_check_read_one_definition_of_the_segmenter(corpus):
    """The anti-drift property itself, asserted rather than left to two call sites agreeing.

    The rule is one sentence — a change that invalidates a cached mask invalidates the committed
    bound too — and it is only true while both read ``segmenter_identity``. A field added to the
    cache key alone would re-open exactly the gap this closes.
    """
    source = corpus["manifest"].parent / "videos" / "ep000.mp4"
    base = FakeMasker().provenance()
    for field in rc.SEGMENTER_IDENTITY_FIELDS:
        moved = dict(base, **{field: "changed"})
        assert rc.MaskCache.key(source, moved) != rc.MaskCache.key(source, base), (
            f"{field} does not reach the mask cache key, so a stale mask would survive it"
        )
        assert rc.segmenter_identity(moved) != rc.segmenter_identity(base), (
            f"{field} does not reach the bound's cross-check, so a stale bound would survive it"
        )


def test_the_committed_bound_is_decided_and_the_refusal_no_longer_says_otherwise():
    """This tripwire fired on 2026-08-26, exactly as it was written to.

    It used to assert ``not AREA_BOUND_ARTIFACT.exists()``, and its docstring said: *the moment
    somebody runs the measure mode and commits a decided bound, this test fails — and whoever made
    it fail should read the refusal's text and check it still describes reality before deleting
    this.* That happened. The refusal's text did NOT still describe reality — it said the
    distribution *HAS NEVER BEEN MEASURED* — and it was repaired rather than the test being
    deleted.

    So the guard is inverted rather than removed, and it now watches the other direction: the
    committed artifact must carry a decided bound WITH a rationale, and the refusal must not
    re-acquire the claim that nothing has been measured. Both halves have failed in production
    before, which is why neither is left to prose.
    """
    assert rc.AREA_BOUND_ARTIFACT.exists(), f"{rc.AREA_BOUND_ARTIFACT} is gone"
    payload = json.loads(rc.AREA_BOUND_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["measurement_qualified"] is True
    assert isinstance(payload["max_frame_fraction"], float)
    assert 0.0 < payload["max_frame_fraction"] < 1.0
    assert payload["bound_rationale"].strip(), "a decided bound with no rationale is a default"

    # Asserted as the POSITIVE claim, not as the absence of the retired phrase. The refusal quotes
    # its own retired sentence in order to retire it out loud, so a `"HAS NEVER BEEN" not in ...`
    # check matches the quotation and fails on a correctly repaired message — the same mistake as
    # asserting `"accuracy" not in text` against a document that says "no accuracy is computed".
    message = rc.area_bound_missing_message(rc.AREA_BOUND_ARTIFACT, "why")
    assert "The distribution IS measured" in message, (
        "the refusal must state that the distribution exists, or it sends the reader to redo it"
    )
    assert str(payload["max_frame_fraction"]) in message, (
        "and it must name the decided bound, so a reader can tell a missing artifact from a "
        "different one"
    )

    bound = rc.load_area_bound()
    assert bound.max_frame_fraction == payload["max_frame_fraction"]
    assert bound.cross_checked is False, (
        "loaded bare, a bound must arrive un-cross-checked so 'forgot to cross-check' is a "
        "refusal rather than a reachable state"
    )


def test_the_measure_mode_reports_the_distribution_and_still_refuses_to_set_the_bound(
    corpus, tmp_path, masker
):
    """The instrument the refusal names, exercised end to end — and it must not close the loop.

    A refusal that names a remedy with no tool behind it is the "plan, not an instrument" failure
    this repo keeps naming. A tool that then PICKS the bound would be the opposite failure: the
    observed maximum cannot fire on the frames it was measured over, and anything above it carries a
    margin nothing in the corpus derives. So the measurement is automated, the decision is not, and
    what it writes must still be refused by the loader.
    """
    record = rc.measure_source_mask_area(corpus["manifest"], masker=masker)

    assert record["max_frame_fraction"] is None
    assert record["measured"]["frames"] == 12 + 9
    assert record["measured"]["empty_frames"] == 0
    assert record["measured"]["max"] == pytest.approx(16 * 16 / (FRAME_H * FRAME_W))
    assert record["prompt"] == rc.ROBOT_TEXT_PROMPT
    assert record["source_manifest_sha256"]

    written = _write(tmp_path / "measured.json", record)
    with pytest.raises(rc.CompositeError, match="has been measured and the bound has not"):
        rc.load_area_bound(written)


@pytest.mark.parametrize(
    "kwargs, reason",
    [({"limit": 1}, "--limit"), ({"stride": 3}, "--stride")],
)
def test_a_truncated_measurement_stamps_itself_as_not_the_corpus(corpus, masker, kwargs, reason):
    """It will not let a smoke test become the bound.

    Both knobs truncate in the direction that matters: fewer episodes and fewer frames can only
    lower the observed maximum, so a bound chosen above a truncated maximum can sit BELOW the real
    one and refuse honest clips — or be nudged up to compensate, which is coining. The artifact
    therefore says so about itself, and load_area_bound reads it.
    """
    record = rc.measure_source_mask_area(corpus["manifest"], masker=masker, **kwargs)

    assert record["measurement_qualified"] is False
    assert any(reason in r for r in record["measurement_disqualified_reasons"])
    assert record["measured"]["episodes_in_manifest"] == 2
    assert record["measured"]["limit"] == kwargs.get("limit")


def test_a_whole_corpus_measurement_at_stride_one_is_qualified(corpus, masker):
    """The other half of the parametrisation above: the honest run must still be usable, or the
    disqualification is just a way of refusing everything."""
    record = rc.measure_source_mask_area(corpus["manifest"], masker=masker)

    assert record["measurement_qualified"] is True
    assert record["measurement_disqualified_reasons"] == []
    assert record["measured"]["limit"] is None and record["measured"]["stride"] == 1


def test_the_measure_mode_exits_three_on_a_smoke_run_so_a_shell_can_tell(corpus, tmp_path, capsys):
    """Exit codes carry meaning here because the shell that runs this cannot parse JSON.

    A zero after a --limit run is what turns a shakedown into "the measurement ran, fine". The
    artifact is still written — a shakedown's numbers are worth reading — and it is stamped.
    """
    out = tmp_path / "smoke_measured.json"
    argv = ["measure", "--manifest", str(corpus["manifest"]), "--out", str(out), "--limit", "1"]

    assert rc.main(argv) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED
    assert rc.EXIT_MEASUREMENT_NOT_QUALIFIED != rc.EXIT_OK != rc.EXIT_REFUSED

    written = json.loads(out.read_text())
    assert written["measurement_qualified"] is False
    assert "NOT THE MEASUREMENT" in capsys.readouterr().err

    with pytest.raises(rc.CompositeError, match="smoke run"):
        rc.load_area_bound(out)


def test_the_measure_mode_counts_empty_frames_because_they_would_refuse_every_clip(corpus, masker):
    """An empty robot mask refuses a clip, so a prompt that leaves source frames empty makes every
    clip containing one refuse. That is a fact worth learning here, before 10 050 clips of GPU
    time, rather than from a chunk of refusals."""
    blind = FakeMasker(rows=(0, 0), cols=(0, 0))
    record = rc.measure_source_mask_area(corpus["manifest"], masker=blind)
    assert record["measured"]["empty_frames"] == 12 + 9
    assert record["measured"]["empty_frame_fraction"] == 1.0


# -- the IoU, which is a diagnostic and never a gate ----------------------------------------------


def test_the_iou_is_computed_correctly_on_a_known_pair():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:4] = True          # 16 px
    b[2:6, 0:4] = True          # 16 px, overlapping rows 2..3 -> 8 px intersection, 24 union
    assert rc.mask_iou(a, b) == pytest.approx(8 / 24)
    assert rc.mask_iou(a, a) == 1.0
    assert rc.mask_iou(a, np.zeros_like(a)) == 0.0
    # Two empty masks agree; a NaN in an artifact reads as "not measured", which is a different fact.
    assert rc.mask_iou(np.zeros_like(a), np.zeros_like(a)) == 1.0


def test_the_iou_is_measured_against_the_raw_generated_frame_and_lands_labelled_as_a_diagnostic(
    corpus, bound, tmp_path
):
    """PR-08 §6 says twice that this number never gates, so the key it lands under says so too.

    It is also measured BEFORE the composite: afterwards the two masks agree by construction and the
    number would be a measurement of robot_composite.py rather than of Cosmos-Transfer2.5. The
    shifted masker makes the two disagree, so a 1.0 here would prove the measurement had been taken
    on the wrong side of the composite.
    """
    context = rc.CompositeContext(
        masker=ShiftedMasker(source_frames=12), bound=_cross_checked(bound, corpus), iou_stride=1,
        cache=None
    )
    generated = _write_clip(tmp_path / "gen_iou.mp4", 12)

    record = context.composite(
        source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",
        generated_video=generated,
    )
    iou = record["robot_mask_iou_source_vs_generated"]

    assert iou["THIS_IS_A_DIAGNOSTIC_ON_THE_GENERATOR_AND_NEVER_A_GATE"] is True
    assert "never as a gate" in iou["note"]
    assert iou["frames_sampled"] == 12
    # rows 4..19 against rows 6..21, 16 columns: intersection 14 rows, union 18.
    assert iou["mean"] == pytest.approx(14 / 18)
    assert iou["min"] == pytest.approx(14 / 18)


def test_no_threshold_is_ever_applied_to_the_iou(corpus, bound, tmp_path):
    """The verdict of a clip must not move when the IoU does. §6 refuses an IoU threshold on this
    mask outright ("would be a coined number"), so a clip whose generated robot is nowhere near the
    real one still composites and still succeeds — with a bad number in the record, which is what
    "diagnostic on the generator" means."""
    disjoint = ShiftedMasker(source_frames=12, gen_shift=20)
    context = rc.CompositeContext(
        masker=disjoint, bound=_cross_checked(bound, corpus), iou_stride=1, cache=None
    )
    generated = _write_clip(tmp_path / "gen_disjoint.mp4", 12)

    record = context.composite(
        source_video=corpus["manifest"].parent / "videos" / "ep000.mp4",
        generated_video=generated,
    )
    assert record["robot_mask_iou_source_vs_generated"]["mean"] == 0.0
    assert record["composited"] is True, "a bad IoU is recorded, never gated on"


# -- the mask cache, and the stale mask it must never serve ---------------------------------------


def test_the_source_mask_is_computed_once_per_source_and_reused_across_restyles(
    corpus, bound, tmp_path, masker
):
    """The mask is a property of the SOURCE frame, so it is identical across all 25 restyles of an
    episode. Recomputing it per restyle would be ~4.3M segmentations instead of ~172k — more
    compute than the generation it protects."""
    context = _context(bound, corpus, cache_dir=tmp_path / "masks", iou_stride=10 ** 6)
    source = corpus["manifest"].parent / "videos" / "ep000.mp4"

    first = context.composite(source_video=source, generated_video=_write_clip(tmp_path / "g1.mp4", 12))
    after_first = masker.calls
    second = context.composite(source_video=source, generated_video=_write_clip(tmp_path / "g2.mp4", 12))

    assert first["mask_source_frames_from_cache"] is False
    assert second["mask_source_frames_from_cache"] is True
    assert masker.calls == after_first + 1, "only the single IoU sample, not another source pass"


def test_a_cache_entry_is_not_reused_when_the_segmenter_changes(corpus, bound, tmp_path, masker):
    """Keying on the path would be the cheap version and it is wrong twice: a regenerated corpus
    reuses paths, and a re-pinned SAM 2 produces different masks from the same bytes. A stale mask
    composites the wrong pixels, and the output still looks like a robot."""
    source = corpus["manifest"].parent / "videos" / "ep000.mp4"
    first_key = rc.MaskCache.key(source, masker.provenance())

    moved = dict(masker.provenance(), version="a-different-pin")
    assert rc.MaskCache.key(source, moved) != first_key

    reprompted = dict(masker.provenance(), prompt="something else.")
    assert rc.MaskCache.key(source, reprompted) != first_key


def test_a_cached_mask_is_still_checked_against_the_bound(corpus, bound, tmp_path):
    """A cache written before the bound was tightened must not skip the check the bound exists to
    make. The refusal has to survive the cache, or the cache is a way past it."""
    # Three quarters of the frame: over the fixture's 0.5 bound and under the loose one. Not the
    # WHOLE frame, because the loose bound would then have to be 1.0 and the loader refuses that —
    # at 1.0 the over-large branch is unreachable, which is a different bug and has its own test.
    fat = FakeMasker(rows=(0, FRAME_H), cols=(0, (FRAME_W * 3) // 4))
    loose = json.loads(bound.read_text())
    loose["max_frame_fraction"] = 0.9
    loose_path = _write(tmp_path / "loose.json", loose)

    source = corpus["manifest"].parent / "videos" / "ep000.mp4"
    cache_dir = tmp_path / "masks"
    permissive = rc.CompositeContext(
        masker=fat, bound=_cross_checked(loose_path, corpus, fat), iou_stride=10 ** 6,
        cache=rc.MaskCache(cache_dir),
    )
    permissive.composite(source_video=source, generated_video=_write_clip(tmp_path / "gA.mp4", 12))

    strict = rc.CompositeContext(
        masker=fat, bound=_cross_checked(bound, corpus, fat), iou_stride=10 ** 6,
        cache=rc.MaskCache(cache_dir),
    )
    with pytest.raises(rc.CompositeError, match="above the committed bound"):
        strict.composite(source_video=source, generated_video=_write_clip(tmp_path / "gB.mp4", 12))


def test_the_recorded_mask_provenance_names_the_robot_prompt_and_not_the_apple_one():
    """The lie this prevents would land in all 10 050 records and be unfalsifiable from them.

    ``apple_sam2.ESTIMATOR_VERSION`` embeds that module's own ``OBJECT_TEXT_PROMPT`` — "apple." —
    because it is written for GEOM_TOL and EST_DRIFT_P95, which segment the apple. These masks are
    of the ROBOT. Copying that string into the G0c record would put ``prompt='apple.'`` beside a
    ``prompt`` field saying ``'robot arm. …'``, with nothing in the artifact able to say which one
    made the mask.
    """
    class StubAdapter:
        ESTIMATOR_NAME = "grounding-dino+sam2+depth-anything-v2"
        ESTIMATOR_VERSION = "det=IDEA/gd@aaa;seg=fb/sam2@bbb;prompt='apple.';box_thr=0.35"
        GROUNDING_DINO_MODEL_CHECKPOINT = "IDEA-Research/grounding-dino-base"
        GROUNDING_DINO_MODEL_REVISION = "a" * 40
        SAM2_MODEL_CHECKPOINT = "facebook/sam2-hiera-large"
        SAM2_MODEL_REVISION = "b" * 40
        BOX_THRESHOLD = 0.15
        TEXT_THRESHOLD = 0.25
        RETRY_BOX_THRESHOLD = 0.1
        RETRY_TEXT_THRESHOLD = 0.1
        # The object-grounding filter's second opinion, named here because the real adapter names
        # it and provenance() refuses a stub that cannot say which predicate decided a detection
        # was the apple (PR-08 V9).
        MASK_VALIDITY_REFERENCE = "warm_saturated_rgb(r>90, r-b>50, saturation>0.35)"

    real = rc.Sam2RobotMasker()
    real._module = StubAdapter()
    prov = real.provenance()

    assert prov["prompt"] == rc.ROBOT_TEXT_PROMPT
    assert "apple" not in prov["version"], "the ROBOT mask's version must not quote the apple prompt"
    assert rc.ROBOT_TEXT_PROMPT in prov["version"]
    assert "b" * 40 in prov["version"], "re-pinning SAM 2 must invalidate every cached mask"
    # The adapter's own string is carried rather than dropped — nothing is hidden, it is labelled.
    assert prov["adapter_version"] == StubAdapter.ESTIMATOR_VERSION
    assert "APPLE prompt" in prov["adapter_version_note"]


def test_the_record_says_that_upstream_s_retry_is_deliberately_not_run():
    """The one divergence from the pinned adapter that a reader could not infer from the mask.

    ``apple_sam2._best_box`` post-processes a second time at (0.10, 0.10) when the first pass
    grounds nothing, because its callers DROP an ungrounded frame and the retry recovers data. G0c
    REFUSES the clip instead, so here the retry would only suppress a refusal — buying a weak box
    that ``_best_box`` itself says can land on something other than the object, i.e. a confident
    wrong robot mask in place of a loud refusal. That makes this masker STRICTER than the segmenter
    PR-08 §4 step 2 pins, which is a real divergence and therefore belongs in the record and in the
    cache key rather than only in a docstring.
    """
    class StubAdapter:
        ESTIMATOR_VERSION = "det=x;seg=y"
        GROUNDING_DINO_MODEL_CHECKPOINT = "IDEA-Research/grounding-dino-base"
        GROUNDING_DINO_MODEL_REVISION = "a" * 40
        SAM2_MODEL_CHECKPOINT = "facebook/sam2-hiera-large"
        SAM2_MODEL_REVISION = "b" * 40
        BOX_THRESHOLD = 0.15
        TEXT_THRESHOLD = 0.25
        RETRY_BOX_THRESHOLD = 0.1
        RETRY_TEXT_THRESHOLD = 0.1
        # The object-grounding filter's second opinion, named here because the real adapter names
        # it and provenance() refuses a stub that cannot say which predicate decided a detection
        # was the apple (PR-08 V9).
        MASK_VALIDITY_REFERENCE = "warm_saturated_rgb(r>90, r-b>50, saturation>0.35)"

    real = rc.Sam2RobotMasker()
    real._module = StubAdapter()
    prov = real.provenance()

    assert "0.1" in prov["upstream_retry_not_run"], "it must name the thresholds it does not use"
    assert "retry=none" in prov["version"], "reversing this must invalidate every cached mask"
    assert "upstream_retry_not_run" in rc.SEGMENTER_IDENTITY_FIELDS

    # And if the adapter stops declaring the retry, the claim stops being checkable — so it refuses
    # rather than writing an unverifiable sentence into 10 050 records.
    del StubAdapter.RETRY_BOX_THRESHOLD
    with pytest.raises(rc.CompositeError, match="no longer declares"):
        rc.Sam2RobotMasker.provenance(real)


# -- the driver's wiring: success implies composited ----------------------------------------------


def test_every_successful_unit_carries_a_g0c_record(corpus, capsys):
    assert rt.main(_argv(corpus)) == 0
    for row in corpus["rows"]:
        record = json.loads((corpus["out"] / row["unit"] / "sample_outputs.json").read_text())
        assert record["status"] == "success"
        g0c = record["g0c"]
        assert g0c["composited"] is True
        assert g0c["frames_composited"] == g0c["frames_total"] == row["frames"]
        assert "unconditionally composited back" in g0c["rule"]
        assert g0c["edge"].startswith("hard binary mask")
        assert g0c["area_bound"]["artifact_sha256"]
        # Per clip, because 97's harvest reads exactly this before filing: the bound that refused
        # over-large masks on THIS clip belonged to the segmenter that made THIS clip's masks.
        assert g0c["area_bound"]["cross_checked"] is True
        assert g0c["area_bound"]["cross_checked_against"]["source_manifest_sha256"]
        assert g0c["robot_mask_iou_source_vs_generated"][
            "THIS_IS_A_DIAGNOSTIC_ON_THE_GENERATOR_AND_NEVER_A_GATE"
        ] is True
    assert "G0c: robot pixels composited back on EVERY frame" in capsys.readouterr().out


def test_a_clip_the_composite_refused_is_never_left_where_the_harvest_can_file_it(corpus, monkeypatch):
    """The harvest files a clip when raw/<unit>/vision.mp4 exists AND the status is success. Those
    are two independent conditions and a refused composite breaks both: PR-08 §6 G0c's claim is that
    the defect CANNOT ENTER, and a claim resting on every future consumer reading a status field
    correctly is weaker than one resting on the file not being there."""
    def refuse(**kw):
        raise rc.CompositeError("the robot mask is EMPTY on frame 0")

    monkeypatch.setattr(rc.CompositeContext, "composite", lambda self, **kw: refuse(**kw))
    assert rt.main(_argv(corpus)) == 0

    for row in corpus["rows"]:
        unit = corpus["out"] / row["unit"]
        assert not (unit / "vision.mp4").exists(), "an uncomposited clip must not keep that name"
        assert (unit / rt.UNCOMPOSITED_QUARANTINE).is_file(), "and its bytes stay, for inspection"
        record = json.loads((unit / "sample_outputs.json").read_text())
        assert record["status"] == "error"
        assert record["g0c"]["composited"] is False


def test_a_stale_quarantine_from_an_earlier_pass_is_withdrawn_before_the_unit_is_retried(corpus):
    unit_dir = corpus["out"] / corpus["rows"][0]["unit"]
    unit_dir.mkdir(parents=True)
    (unit_dir / rt.UNCOMPOSITED_QUARANTINE).write_bytes(b"from a previous pass")

    assert rt.main(_argv(corpus)) == 0
    assert not (unit_dir / rt.UNCOMPOSITED_QUARANTINE).exists()
    assert (unit_dir / "vision.mp4").is_file()


def test_a_missing_area_bound_stops_the_whole_run_before_the_first_unit(corpus):
    """Run-level facts refuse at run level. Discovered per unit, a missing bound would become N
    identical errors that read like a flaky generator and spend a pass of the chunk's rail."""
    assert rt.main(_argv(corpus, {"--robot-mask-area-bound": str(corpus["root"] / "nope.json")})) == 1
    assert not corpus["out"].exists() or not any(corpus["out"].glob("*/vision.mp4"))


def test_unstaged_segmentation_checkpoints_stop_the_run_rather_than_every_unit(corpus, capsys):
    """Same argument for the weights: apple_sam2 refuses loudly when its pinned checkpoints are not
    cached, and that refusal belongs before the first clip — and as a FATAL line, not a traceback.

    The real masker's failure is an ``EstimatorDependencyMissing``, an ImportError subclass carrying
    a multi-paragraph diagnosis. Escaping ``main`` it would print as a traceback, which the sbatch
    reads as "the driver crashed" rather than as a refusal it can act on. This uses the real
    Sam2RobotMasker with a stubbed estimator module so that the translation is exercised rather than
    described.
    """
    class NoWeights:
        def _detector(self):
            raise ImportError("no SAM 2 checkpoint at its pinned revision e6a8e880")

    real = rc.Sam2RobotMasker()
    real._module = NoWeights()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rc, "build_masker", lambda: real)
        assert rt.main(_argv(corpus)) == 1
    err = capsys.readouterr().err
    assert err.startswith("FATAL:")
    assert "pinned revision e6a8e880" in err, "the estimator's own diagnosis must survive intact"
    assert not any(corpus["out"].glob("*/vision.mp4"))


# -- the one that matters: there is no path that skips the composite ------------------------------


def test_run_unit_cannot_be_called_without_a_composite_context():
    """Not optional with a None default, not read from a flag. A caller that has no context cannot
    call the function at all, which is a stronger guarantee than any check inside it."""
    parameter = inspect.signature(rt.run_unit).parameters["composite"]
    assert parameter.default is inspect.Parameter.empty


def test_no_way_to_build_a_context_that_does_not_composite():
    """build_context takes where the inputs come from and how often the DIAGNOSTIC is sampled. It
    takes nothing that decides WHETHER to composite, and build_masker takes nothing at all.

    ``source_manifest`` is in the list and is not an input to the composite: it is the corpus the
    committed area bound is cross-checked against. It has no default, deliberately — an optional
    cross-check is a cross-check that gets skipped.
    """
    signature = inspect.signature(rc.build_context)
    assert set(signature.parameters) == {
        "source_manifest", "area_bound_path", "iou_stride", "cache_dir", "preflight"
    }
    assert signature.parameters["source_manifest"].default is inspect.Parameter.empty
    assert not inspect.signature(rc.build_masker).parameters


def test_no_driver_flag_can_turn_the_composite_off():
    """The cheapest way for "unconditional" to stop being true is somebody adding --skip-composite
    for a debugging session in six months and leaving it in. This walks the parser instead of
    trusting a review to notice."""
    forbidden = re.compile(r"skip|no[-_]?composit|disable|without|bypass|raw[-_]?only", re.I)
    parser = rt.build_parser()
    for action in parser._actions:
        for option in action.option_strings:
            assert not forbidden.search(option), f"{option} looks like a way past G0c"
        assert "composit" not in (action.dest or ""), (
            f"--{action.dest} names the composite. Nothing on this command line may: the three G0c "
            "options choose where its INPUTS come from and how often the diagnostic is sampled."
        )
    # And the surface it IS allowed is exactly these three, so a fourth has to be argued for.
    g0c_options = {d for d in (a.dest for a in parser._actions)
                   if d in {"robot_mask_area_bound", "mask_cache", "iou_stride"}}
    assert g0c_options == {"robot_mask_area_bound", "mask_cache", "iou_stride"}


def test_the_composite_runs_on_every_backend_including_the_placeholder(corpus, masker):
    """--backend null is a placeholder GENERATOR, not a placeholder pipeline. Exempting it would
    have created the one thing G0c must not have: a reachable code path that skips compositing."""
    assert rt.main(_argv(corpus)) == 0
    assert masker.calls > 0
    for row in corpus["rows"]:
        record = json.loads((corpus["out"] / row["unit"] / "sample_outputs.json").read_text())
        assert record["backend"] == "null"
        assert record["g0c"]["composited"] is True


def test_the_sbatch_invokes_the_driver_with_no_flag_this_pipeline_does_not_know():
    """The production command line, read out of 97_transfer25_restyle.sbatch rather than remembered.

    Two things are asserted about it. It must not pass --backend, because the default is transfer25
    and the null placeholder must stay unreachable from the cluster path. And every flag it does
    pass has to be one this driver's parser declares, so a rename here cannot silently leave the
    generation job running the old contract.
    """
    sbatch = pathlib.Path(__file__).resolve().parents[1] / "cluster/discoverer/97_transfer25_restyle.sbatch"
    text = sbatch.read_text(encoding="utf-8")
    # One backslash-continued command per invocation — the timing path and the generation path.
    calls = re.findall(r'python "\$\{RESTYLE_DRIVER\}"((?:[^\n]*\\\n)*[^\n]*)', text)
    assert len(calls) == 2, f"expected the timing and generation invocations, found {len(calls)}"

    declared = {o for a in rt.build_parser()._actions for o in a.option_strings}
    for call in calls:
        flags = {f for f in re.findall(r"\s(--[a-z0-9]+(?:-[a-z0-9]+)*)\b", call)}
        assert flags <= declared, (
            f"the sbatch passes {sorted(flags - declared)}, which the driver does not take"
        )
        assert "--backend" not in flags, (
            "the sbatch must not select a backend: the default is transfer25, and --backend null is "
            "the placeholder generator. It stays unreachable from the cluster path."
        )


# ================================================================================================
# 97's harvest — the one place G0c becomes a claim about a directory
# ================================================================================================
#
# chunk_metadata.json and the NOT_TRAINING_DATA marker both state that the composite ran on every
# clip filed into clips/<style_set>/. Until 2026-08-22 that was prose: the harvest filed on two
# conditions that can both hold without it (vision.mp4 exists, status == success) and never read a
# thing the compositor wrote. PR-08 §6 calls G0c "solved by construction", and a construction that
# is asserted rather than verified is not one — it is the same shape as PR-07's -359.41, a number
# whose instrument nobody checked.
#
# These tests RUN THE SBATCH'S OWN HARVEST, extracted from the file, rather than a copy of it here.
# A copy would pass forever after the sbatch changed.

SBATCH_97 = pathlib.Path(__file__).resolve().parents[1] / "cluster/discoverer/97_transfer25_restyle.sbatch"


def _harvest_source() -> str:
    """The python heredoc inside 97's ``harvest()``, verbatim. Refuses rather than guessing."""
    text = SBATCH_97.read_text(encoding="utf-8")
    start = text.index("harvest() {")
    body = text[start:]
    opener = body.index("<<'PY'\n") + len("<<'PY'\n")
    end = body.index("\nPY\n", opener)
    source = body[opener:end]
    assert "g0c" in source, "97's harvest no longer reads the compositor's record at all"
    return source


def _run_harvest(tmp_path: pathlib.Path, units: list[dict]) -> "subprocess.CompletedProcess":
    """Lay out a raw dir the way the driver would, then run 97's harvest over it."""
    import subprocess

    raw, dest = tmp_path / "_raw", tmp_path / "clips"
    raw.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    work = tmp_path / "work.jsonl"
    rows = []
    for unit in units:
        name = unit["unit"]
        (raw / name).mkdir(parents=True, exist_ok=True)
        if unit.get("video", True):
            (raw / name / "vision.mp4").write_bytes(b"not decoded by the harvest")
        if unit.get("record") is not None:
            _write(raw / name / "sample_outputs.json", unit["record"])
        rows.append(json.dumps({"unit": name, "frames": unit["frames"]}))
    work.write_text("\n".join(rows) + "\n", encoding="utf-8")

    script = tmp_path / "harvest_from_97.py"
    script.write_text(_harvest_source(), encoding="utf-8")
    # argv[5] is the REFUSED list, added 2026-08-28. The harvest now tells a unit G0c refused on its
    # source masks (permanent, never retried, never filed) apart from one that is simply absent
    # (retried, and bounded by the no-op guard). Before that split a refused unit sat on the missing
    # list forever and the chunk exited 1 without ever writing its PR-08 §6 record.
    return subprocess.run(
        [sys.executable, str(script), str(work), str(raw), str(dest), str(tmp_path / "missing"),
         str(tmp_path / "refused.json")],
        capture_output=True, text=True,
    )


def _composited_record(frames: int, **over) -> dict:
    """What restyle_transfer25 writes for a unit that succeeded — the shape the harvest reads."""
    record = {
        "status": "success", "backend": "transfer25", "seed": 7001,
        "episode": "ep000", "style": "train-01",
        "g0c": {
            "composited": True, "frames_composited": frames, "frames_total": frames,
            "area_bound": {"cross_checked": True, "artifact_sha256": "a" * 64,
                           "max_frame_fraction": 0.5},
            "masker": {"prompt": rc.ROBOT_TEXT_PROMPT},
        },
    }
    record.update(over)
    return record


def test_the_harvest_files_a_clip_only_with_its_own_g0c_evidence_beside_it(tmp_path):
    """The evidence has to travel with the clip: the raw record can be pruned, and the claim in
    chunk_metadata.json is about the CLIPS directory."""
    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12)}])

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "clips" / "u1.mp4").is_file()
    proof = json.loads((tmp_path / "clips" / "u1.g0c.json").read_text())
    assert proof["g0c"]["composited"] is True
    assert proof["frames"] == 12
    assert "frames_composited==frames_total==work_list.frames" in proof["verified"]


def test_the_harvest_refuses_a_success_that_carries_no_composite(tmp_path):
    """A driver that reports success without compositor evidence is not the driver 97 describes,
    and filing its output would make the chunk record's strongest sentence false."""
    record = _composited_record(12)
    del record["g0c"]
    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": record}])

    assert result.returncode != 0
    assert "no g0c.composited" in result.stdout + result.stderr
    assert not (tmp_path / "clips" / "u1.mp4").exists()


def test_the_harvest_refuses_a_clip_composited_on_only_some_of_its_frames(tmp_path):
    """The frame count is compared against the WORK-LIST ROW, which is the per-clip, specific half:
    a record copied from another unit, or a composite that stopped early, disagrees with it."""
    record = _composited_record(12)
    record["g0c"]["frames_composited"] = 11
    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": record}])

    assert result.returncode != 0
    assert "composited 11 of 12" in result.stdout + result.stderr


def test_the_harvest_refuses_a_bound_that_was_never_cross_checked(tmp_path):
    """The area bound is what refuses an over-large mask. One measured under other weights or over
    another corpus is a number with no distribution behind it, and a clip refused by such a bound
    was not really checked."""
    record = _composited_record(12)
    record["g0c"]["area_bound"]["cross_checked"] = False
    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": record}])

    assert result.returncode != 0
    assert "cross-checked" in result.stdout + result.stderr


def test_the_harvest_refuses_a_placeholder_backend_clip_by_content(tmp_path):
    """--backend null writes a REAL mp4 since G0c landed, so nothing about the file keeps a
    placeholder out of a corpus. 97 never passes --backend; this is what makes that a property of
    the tree rather than a habit of the submit script."""
    result = _run_harvest(
        tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12, backend="null")}]
    )

    assert result.returncode != 0
    assert "transfer25" in result.stdout + result.stderr


def test_a_failed_unit_goes_to_the_missing_list_rather_than_stopping_the_chunk(tmp_path):
    """The refusals above are fatal because they mean the generation path is wrong. An ordinary
    failed unit is not: it is what the missing-list and the requeue exist for, and turning it into
    a fatal would stop a chunk over one bad clip."""
    result = _run_harvest(tmp_path, [
        {"unit": "u1", "frames": 12, "record": _composited_record(12)},
        {"unit": "u2", "frames": 9, "record": {"status": "error", "backend": "transfer25",
                                               "g0c": {"composited": False}}},
    ])

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "clips" / "u1.mp4").is_file()
    assert not (tmp_path / "clips" / "u2.mp4").exists()
    assert "u2" in (tmp_path / "missing").read_text()


def test_a_clip_filed_by_an_earlier_driver_cannot_inherit_the_new_claim(tmp_path):
    """The requeue path. The old harvest skipped any unit already in clips/, so a clip filed before
    G0c existed would survive every later pass untouched and be covered by a record that says the
    composite ran on everything in the directory. Its raw record may be long gone, so the sidecar
    is the only durable per-clip proof and its ABSENCE is the refusal."""
    (tmp_path / "clips").mkdir(parents=True)
    (tmp_path / "clips" / "u1.mp4").write_bytes(b"filed before G0c existed")

    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12)}])

    assert result.returncode != 0
    assert "already filed with no u1.g0c.json" in result.stdout + result.stderr


def test_a_clip_outside_this_chunk_s_work_list_is_still_covered(tmp_path):
    """The per-unit checks above can only see units in the work list, and a clip that is not in it
    is exactly where a false claim would hide. So the invariant is re-checked over the whole
    directory before the harvest returns."""
    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12)}])
    assert result.returncode == 0, result.stderr

    (tmp_path / "clips" / "stranger.mp4").write_bytes(b"from somewhere else")
    again = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12)}])

    assert again.returncode != 0
    assert "no G0c evidence file" in again.stdout + again.stderr


def test_an_evidence_file_that_proves_nothing_is_not_evidence(tmp_path):
    """The sidecar is READ, not counted, and this is the reason in the sbatch's own words.

    97 already records the rule for the raw status file — "a harvest keyed on file presence counts
    failures as finished work" — and a sidecar checked only with ``is_file()`` is that same check
    under another name. An empty ``{}`` beside a clip satisfies ``g0c_evidence.evidence_files ==
    clips`` in chunk_metadata.json while proving nothing about whether the composite ran, which is
    exactly the claim-without-an-instrument shape this whole check exists to end.
    """
    (tmp_path / "clips").mkdir(parents=True)
    (tmp_path / "clips" / "u1.mp4").write_bytes(b"filed with a sidecar that says nothing")
    _write(tmp_path / "clips" / "u1.g0c.json", {})

    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12)}])

    assert result.returncode != 0
    assert "carries no g0c.composited: true" in result.stdout + result.stderr


def test_an_evidence_file_belonging_to_a_different_clip_is_refused(tmp_path):
    """A sidecar copied from another unit would pass a presence check and a content check that only
    asked "does this say composited". The work-list row is the one thing that knows how many frames
    THIS clip has, so the frame count is compared here — the directory-wide sweep cannot do it for
    a unit that has no row."""
    (tmp_path / "clips").mkdir(parents=True)
    (tmp_path / "clips" / "u1.mp4").write_bytes(b"filed")
    _write(tmp_path / "clips" / "u1.g0c.json",
           {"unit": "u1", "frames": 99, "g0c": {"composited": True}})

    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": _composited_record(12)}])

    assert result.returncode != 0
    assert "records 99 frames" in result.stdout + result.stderr


def test_a_frame_count_that_is_not_a_number_is_refused_rather_than_crashing(tmp_path):
    """``int(None)`` is a TypeError and a traceback is not a refusal — the operator reads it as this
    job crashing rather than as a diagnosis of the record. A frame count that is null, absent or a
    string proves nothing about how many frames were composited, so it takes the same path as one
    that disagrees."""
    record = _composited_record(12)
    record["g0c"]["frames_composited"] = None

    result = _run_harvest(tmp_path, [{"unit": "u1", "frames": 12, "record": record}])

    assert result.returncode != 0
    assert "Traceback" not in result.stderr, "a refusal, not a crash"
    assert "cannot be filed" in result.stdout + result.stderr


def test_a_requeue_over_a_directory_this_harvest_filed_is_idempotent(tmp_path):
    """The refusals above must not turn a normal second pass into a fatal: 97 harvests before every
    pass and after the driver returns, so a clip this harvest filed is seen again by design."""
    units = [{"unit": "u1", "frames": 12, "record": _composited_record(12)}]
    first = _run_harvest(tmp_path, units)
    assert first.returncode == 0, first.stderr

    again = _run_harvest(tmp_path, units)
    assert again.returncode == 0, again.stdout + again.stderr
    assert (tmp_path / "clips" / "u1.g0c.json").is_file()


def test_the_sbatch_clears_the_detection_thresholds_a_submit_environment_could_smuggle_in(tmp_path):
    """apple_sam2 reads BOX_THRESHOLD / TEXT_THRESHOLD and its retry pair from the environment,
    deliberately, because a measurement sweep needs to move them. The G0c robot mask is made
    through that same adapter, and there a threshold decides WHICH PIXELS THE GENERATOR MAY TOUCH:
    lower grows the mask, higher shrinks it, and a smaller mask is generated manipulator left in
    the frame. robot_composite refuses a per-run prompt for that reason and cannot refuse these,
    because the adapter owns them — so 97 clears them, before either driver invocation."""
    text = SBATCH_97.read_text(encoding="utf-8")
    for variable in ("WAM_PR08_BOX_THRESHOLD", "WAM_PR08_TEXT_THRESHOLD",
                     "WAM_PR08_RETRY_BOX_THRESHOLD", "WAM_PR08_RETRY_TEXT_THRESHOLD",
                     "WAM_PR08_OBJECT_PROMPT"):
        assert re.search(rf"^unset .*\b{variable}\b", text, re.M), (
            f"{variable} reaches the pinned segmenter from the environment and 97 does not clear it"
        )
        assert text.index(f"unset") < text.index('python "${RESTYLE_DRIVER}"'), (
            "clearing them after the driver has run clears nothing"
        )


def test_every_g0c_claim_97_makes_about_its_clips_directory_names_the_evidence_for_it():
    """The regression this guards is a sentence, not a code path, and it is the expensive kind.

    Three places in 97 state that the composite ran on every clip filed: the header's (b), the
    NOT_TRAINING_DATA marker and chunk_metadata.json. Each of them is only true because harvest()
    reads the compositor's record and writes <unit>.g0c.json beside the clip. A future edit that
    keeps the sentence and drops the mechanism is exactly what a reviewer caught here once.
    """
    text = SBATCH_97.read_text(encoding="utf-8")
    assert "g0c.json" in text, "nothing in 97 writes or names the per-clip evidence file any more"
    assert '"g0c_evidence"' in text, "chunk_metadata.json no longer counts the evidence it claims"
    assert "len(g0c_proofs) != n_clips" in text, (
        "the chunk record no longer refuses to be written when the directory cannot support its "
        "own G0c claim"
    )


# -- G0c, asked BEFORE the generator instead of only after it -------------------------------------
#
# The defect these cover cost half a GPU-hour per attempt and produced nothing. The masks come from
# the SOURCE video, so every refusal check_mask can raise is knowable before the backend is called;
# until the preflight landed, the driver generated the full clip first and refused on frame 0
# afterwards. 385 of the corpus's 402 episodes are refused by one half of that check or the other
# (runs/pr08-robot-mask-area/POOLED.json), and the episode 97's TIMING=1 path times is one of them.


def _same_episode_work_list(corpus, styles=("train-01", "train-02")) -> list[dict]:
    """Two units over ONE episode, which is the shape stage 1 actually has.

    One invocation of the driver generates one ``--style-set``, so the repeats it can re-discover a
    refusal over are the ones within that set: four per episode under T40_RULE_V11 §2's stage 1
    (its 8 style-instances are 4 train plus 4 matched identity repeats, submitted as two jobs), ten
    per episode per set in the full rendering. Two is enough to tell "discovered once" from
    "discovered per unit", which is the whole distinction.
    """
    rows = [
        {"unit": f"ep000__{style}__r00", "episode": "ep000", "frames": 12,
         "style": style, "repeat": 0, "seed": 7100 + i}
        for i, style in enumerate(styles)
    ]
    corpus["work"].write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def test_the_source_mask_preflight_refuses_the_unit_before_the_backend_is_ever_called(
    corpus, monkeypatch
):
    """The whole point. A unit G0c was always going to refuse must not be generated first.

    The backend is replaced with a recorder rather than merely observed, because "was it called"
    is the entire claim: the refusal itself was already correct before the preflight existed, it
    just arrived after ~11 minutes of H200 time and one of the four running-job slots.
    """
    real = rt._null_backend
    called: list[str] = []

    def recording(sample, out_dir):
        called.append(sample["name"])
        return real(sample, out_dir)

    monkeypatch.setattr(rt, "_null_backend", recording)
    monkeypatch.setattr(rc, "build_masker", lambda: FakeMasker(rows=(0, 0), cols=(0, 0)))

    # Still 0: a per-unit refusal is per-unit. One episode's masks say nothing about the next's.
    assert rt.main(_argv(corpus)) == 0
    assert called == [], f"the generator ran on units G0c refuses: {called}"

    for row in corpus["rows"]:
        unit = corpus["out"] / row["unit"]
        record = json.loads((unit / "sample_outputs.json").read_text())
        assert record["status"] == "error"
        assert record["detail"].startswith("SourceMaskRefusal:")
        assert "EMPTY on frame 0" in record["detail"]
        assert record["g0c"]["composited"] is False
        assert record["g0c"]["refused_before_generation"] is True
        assert not (unit / "vision.mp4").exists()
        # Nothing was generated, so there is nothing to quarantine — and the absence is the saving.
        assert not (unit / rt.UNCOMPOSITED_QUARANTINE).exists()


def test_the_preflight_refuses_an_over_large_mask_before_the_backend_too(corpus, monkeypatch):
    """Both halves of check_mask, not just the empty one.

    The preflight calls check_mask itself rather than re-implementing either half, so this is a
    test that the WHOLE predicate moved and not the branch that happened to be in front of us: an
    over-large mask means the composite copies the source back over everything, the restyle becomes
    a no-op, and arms B and C silently become arm A at full GPU cost.
    """
    real = rt._null_backend
    called: list[str] = []
    monkeypatch.setattr(rt, "_null_backend", lambda s, d: (called.append(s["name"]), real(s, d))[1])
    monkeypatch.setattr(rc, "build_masker", lambda: FakeMasker(rows=(0, FRAME_H), cols=(0, FRAME_W)))

    assert rt.main(_argv(corpus)) == 0
    assert called == []
    record = json.loads((corpus["out"] / corpus["rows"][0]["unit"] / "sample_outputs.json").read_text())
    assert "above the committed bound" in record["detail"]
    assert record["g0c"]["refused_before_generation"] is True


def test_a_unit_the_preflight_passes_is_still_composited_and_checked_frame_by_frame(corpus):
    """The preflight moves the discovery, never the decision.

    A unit that clears it is generated, composited and checked again by composite_clip over every
    frame — the post-composite check is untouched, and the per-clip record still carries the
    evidence 97's harvest refuses to file a clip without.
    """
    assert rt.main(_argv(corpus)) == 0
    for row in corpus["rows"]:
        record = json.loads((corpus["out"] / row["unit"] / "sample_outputs.json").read_text())
        assert record["status"] == "success"
        pre = record["g0c_source_mask_preflight"]
        assert pre["checked"] is True and pre["frames_checked"] == row["frames"]
        assert record["g0c"]["composited"] is True
        assert record["g0c"]["frames_composited"] == record["g0c"]["frames_total"] == row["frames"]


def test_the_preflight_and_the_composite_run_one_mask_pass_between_them_not_two(
    corpus, tmp_path, masker
):
    """The measurand 97's TIMING=1 window is around must not move, and it does not.

    The preflight computes the source masks through robot_composite.source_masks, so they land in
    the same MaskCache the composite then reads: one pass per source episode before this change and
    one after it. That matters beyond tidiness — the timed window in 97 wraps the whole driver, and
    a second mask pass inside it would inflate seconds_per_frame, which is the number PR-08 §8
    item 3 derives the partition's GPU-h ceiling from.
    """
    argv = _argv(corpus, {"--mask-cache": str(tmp_path / "masks"), "--iou-stride": "1000000"})
    assert rt.main(argv) == 0
    # 12 + 9 source frames, one pass each, plus the single IoU sample per unit (index 0 always
    # lands under any stride). A second source pass would show up here as +21.
    assert masker.calls == 12 + 9 + 2, masker.calls


def test_a_refusal_is_not_re_walked_for_every_style_instance_of_the_same_episode(
    corpus, monkeypatch, capsys
):
    """Defect 11: eight style-instances over one refused episode must cost one mask pass, not eight.

    The instrument is source_masks rather than the masker's call count, because the MaskCache alone
    would already make the second unit's masker calls zero — and that is exactly the confusion
    worth ruling out. The cache makes re-discovery cheap; the memo makes it not happen.
    """
    rows = _same_episode_work_list(corpus)
    monkeypatch.setattr(rc, "build_masker", lambda: FakeMasker(rows=(0, 0), cols=(0, 0)))
    real_source_masks = rc.source_masks
    passes: list[str] = []

    def counting(source_video, frames, context):
        passes.append(str(source_video))
        return real_source_masks(source_video, frames, context)

    monkeypatch.setattr(rc, "source_masks", counting)

    assert rt.main(_argv(corpus)) == 0
    assert len(passes) == 1, f"the same source was walked {len(passes)} times: {passes}"

    # And NOT silently: both units carry their own error record, with the reason.
    for row in rows:
        record = json.loads((corpus["out"] / row["unit"] / "sample_outputs.json").read_text())
        assert record["status"] == "error"
        assert "EMPTY on frame 0" in record["detail"]
        assert record["g0c"]["refused_before_generation"] is True
    assert "Remembered from an earlier unit of this run" in json.loads(
        (corpus["out"] / rows[1]["unit"] / "sample_outputs.json").read_text()
    )["detail"]

    out = capsys.readouterr().out
    assert "2 of 2 units refused BEFORE generation" in out
    assert "1 distinct sources" in out and "1 of the refusals were served from this run's memo" in out


def test_the_memo_never_remembers_that_an_episode_PASSED(corpus, monkeypatch):
    """Only refusals are memoised, and that asymmetry is the safety property.

    A memo of passes would be a way past G0c the moment any input changed in a way the key failed
    to capture; a memo of refusals can at worst refuse a unit G0c also refuses. So a passing
    episode is re-checked for every style-instance of it — at the price of a cache read, which is
    what the cache is for.
    """
    _same_episode_work_list(corpus)
    real_check = rc.check_mask
    checks: list[int] = []

    def counting(mask, *, frame_index, bound, source):
        checks.append(frame_index)
        return real_check(mask, frame_index=frame_index, bound=bound, source=source)

    monkeypatch.setattr(rc, "check_mask", counting)
    assert rt.main(_argv(corpus)) == 0
    # Two units x 12 frames x (one preflight + one post-composite pass). The memo saves none of
    # them, because none of them refused.
    assert len(checks) == 2 * 12 * 2, len(checks)


def test_the_memo_key_cannot_leak_across_a_different_bound_or_a_different_segmenter(
    corpus, bound, tmp_path
):
    """The key is the predicate's own inputs: the source bytes, the segmenter, and the bound.

    A memo keyed on the episode id would survive exactly the changes it must not survive — a
    re-pinned SAM 2 produces different masks from the same bytes, and a re-decided bound is a
    different question about the same masks. Both are checked here against the same source file, so
    the only thing that can be moving the key is the thing under test.
    """
    source = corpus["manifest"].parent / "videos" / "ep000.mp4"
    base = rc.CompositeContext(
        masker=FakeMasker(), bound=_cross_checked(bound, corpus), iou_stride=10, cache=None
    )
    baseline = rt.SourceMaskMemo.key(source, base)

    class Repinned(FakeMasker):
        def provenance(self) -> dict:
            return dict(super().provenance(), version="a-different-pin")

    repinned = rc.CompositeContext(
        masker=Repinned(), bound=base.bound, iou_stride=10, cache=None
    )
    assert rt.SourceMaskMemo.key(source, repinned) != baseline

    loose = json.loads(bound.read_text())
    loose["max_frame_fraction"] = 0.9
    loose_path = _write(tmp_path / "loose.json", loose)
    moved_bound = rc.CompositeContext(
        masker=FakeMasker(), bound=_cross_checked(loose_path, corpus), iou_stride=10, cache=None
    )
    assert rt.SourceMaskMemo.key(source, moved_bound) != baseline

    # And it is stable for the thing it is supposed to be stable for: same source, same segmenter,
    # same bound, a different unit of the same run.
    twin = rc.CompositeContext(
        masker=FakeMasker(), bound=_cross_checked(bound, corpus), iou_stride=10, cache=None
    )
    assert rt.SourceMaskMemo.key(source, twin) == baseline


def test_the_preflight_reads_the_same_bound_object_the_composite_will_use(corpus, monkeypatch):
    """Exactness, asserted rather than argued: one bound, one predicate, one mask source.

    If the preflight could ever pass a unit the composite then refuses it would be worse than
    nothing — the operator would have paid for the generation AND for a refusal that claimed to
    have been pre-checked. So the bound the preflight passes to check_mask is asserted to be the
    context's own frozen AreaBound, by identity.
    """
    seen: list[object] = []
    real_check = rc.check_mask

    def watching(mask, *, frame_index, bound, source):
        seen.append(bound)
        return real_check(mask, frame_index=frame_index, bound=bound, source=source)

    monkeypatch.setattr(rc, "check_mask", watching)
    assert rt.main(_argv(corpus)) == 0
    assert seen, "check_mask was not called at all"
    assert all(b is seen[0] for b in seen), "two different bounds reached one run's checks"


def test_the_timing_path_now_refuses_in_seconds_instead_of_after_a_generated_clip(
    corpus, monkeypatch
):
    """97's TIMING=1 contract, end to end, on the shape the corpus actually has.

    The timed unit is `head -1` of a deterministically sorted work list, and 385 of 402 episodes
    are refused by one half of check_mask or the other — so the timing run's most likely outcome is
    a refusal. It was previously a refusal that arrived after the full clip had been generated:
    ~0.3-0.5 GPU-h and one of four running-job slots, per attempt, for no THROUGHPUT.json. The exit
    code is unchanged (--require-success -> 1, and 97 writes no artifact); what changed is that
    nothing was generated to reach it.
    """
    real = rt._null_backend
    called: list[str] = []
    monkeypatch.setattr(rt, "_null_backend", lambda s, d: (called.append(s["name"]), real(s, d))[1])
    monkeypatch.setattr(rc, "build_masker", lambda: FakeMasker(rows=(0, 0), cols=(0, 0)))

    assert rt.main(_argv(corpus, {"--require-success": None})) == 1
    assert called == [], "the generator ran before the refusal that was knowable without it"
    assert not list(corpus["out"].glob("*/vision.mp4"))


class CountingFilterMasker(FakeMasker):
    """A masker that carries PR-08 V9's cumulative ``filter_counters``, as the real one does.

    ``FakeMasker`` declares none, so ``composite_clip``'s ``getattr(..., {})`` sees an empty dict
    and every object-filter assertion in this file is vacuous. The real ``Sam2RobotMasker`` counts
    six things and never resets them; the caller that brackets a mask pass is the only thing that
    can say what that pass did. This declares two of the six, incremented per masked frame, which
    is enough to tell a differenced count from a zero.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.filter_counters = {"frames_masked": 0, "detections_dropped_as_object": 0}

    def mask(self, rgb) -> np.ndarray:
        self.filter_counters["frames_masked"] += 1
        self.filter_counters["detections_dropped_as_object"] += 1
        return super().mask(rgb)


def test_the_preflight_records_the_object_filter_counts_its_own_mask_pass_produced(
    corpus, monkeypatch, tmp_path
):
    """Moving the mask pass moved PR-08 V9's counters with it, and they must land somewhere.

    ``composite_clip`` differences ``masker.filter_counters`` around its own ``source_masks`` call,
    because — robot_composite's own words — "a filter whose firing is not recorded cannot be told
    apart from a corpus that never triggered it". Once the preflight computes the masks first, that
    call is a cache hit for EVERY clip the driver produces, its delta is zero for every clip, and
    the corpus's record loses the only number that separates "the segmenter found no robot" from
    "the filter removed the only detection" — which is the question T40_RULE_V12's empty-mask
    semantics turn on. The composite's block still reads zero here, with masks_from_cache true, and
    that is correct and documented; what this asserts is that the counts exist in the block that
    ran the pass.
    """
    masker = CountingFilterMasker()
    monkeypatch.setattr(rc, "build_masker", lambda: masker)
    assert rt.main(_argv(corpus, {"--mask-cache": str(tmp_path / "masks")})) == 0

    for row in corpus["rows"]:
        record = json.loads((corpus["out"] / row["unit"] / "sample_outputs.json").read_text())
        counted = record["g0c_source_mask_preflight"]["robot_mask_object_filter"]
        assert counted["masks_from_cache"] is False
        assert counted["frames_masked"] == row["frames"], counted
        assert counted["detections_dropped_as_object"] == row["frames"], counted
        # And the composite's own copy is the cache-hit zero the note describes, so the two blocks
        # never both claim the same firings.
        composited = record["g0c"]["robot_mask_object_filter"]
        assert composited["masks_from_cache"] is True
        assert composited["frames_masked"] == 0
