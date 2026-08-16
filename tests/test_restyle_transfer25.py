"""The restyle driver's refusals, and the seed channel arm C depends on.

These run entirely on the null backend: no GPU, no Transfer2.5 checkout, no weights. What they
exercise is everything AROUND the model call, which is where PR-08's controls actually live — the
seed reaching the sampler, the per-unit isolation, the status file being written after the mp4 is
asserted, and the four things the driver refuses to default.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import restyle_transfer25 as rt  # noqa: E402

STYLE_SET = "train"


def _write(path: pathlib.Path, payload) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def corpus(tmp_path: pathlib.Path):
    """A two-episode source, a two-style partition, and a work list over them."""
    src = tmp_path / "source"
    for ep in ("ep000", "ep001"):
        (src / "videos").mkdir(parents=True, exist_ok=True)
        (src / "videos" / f"{ep}.mp4").write_bytes(b"fake mp4")
    manifest = _write(
        src / "manifest.json",
        {
            "resolution": [640, 480],
            "episodes": [
                {"id": "ep000", "frames": 120, "video": "videos/ep000.mp4"},
                {"id": "ep001", "frames": 130, "video": "videos/ep001.mp4"},
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
        {"unit": "ep000__train-01__r00", "episode": "ep000", "frames": 120,
         "style": "train-01", "repeat": 0, "seed": 7001},
        {"unit": "ep001__train-02__r00", "episode": "ep001", "frames": 130,
         "style": "train-02", "repeat": 0, "seed": 7002},
    ]
    work = tmp_path / "work.jsonl"
    work.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"manifest": manifest, "styles": styles, "work": work, "out": tmp_path / "raw", "rows": rows}


def _argv(corpus, **over) -> list[str]:
    args = {
        "--checkpoint-path": "/staged/ckpt",
        "--manifest": str(corpus["manifest"]),
        "--styles": str(corpus["styles"]),
        "--style-set": STYLE_SET,
        "--work-list": str(corpus["work"]),
        "--out": str(corpus["out"]),
        "--control": "depth:0.5",
        "--backend": "null",
    }
    args.update(over)
    flat = [x for kv in args.items() for x in kv]
    return flat + ["--no-guardrails"]


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
        rt.WorkUnit("u", "ep000", 120, "train-01", 0, 7001),
        source_root=corpus["manifest"].parent,
        episode={"id": "ep000", "frames": 120, "video": "videos/ep000.mp4"},
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
        json.dumps({"unit": "u", "episode": "ep000", "frames": 120, "style": "train-01", "repeat": 0})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rt.DriverError, match="seed"):
        rt.load_work_list(corpus["work"])


def test_a_non_integer_seed_is_refused(corpus):
    corpus["work"].write_text(
        json.dumps({"unit": "u", "episode": "ep000", "frames": 120, "style": "train-01",
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
        rt.WorkUnit("u", "ep000", 120, "train-01", 0, 7001),
        source_root=src,
        episode={"id": "ep000", "frames": 120, "video": "videos/ep000.mp4", "depth": "depth/ep000.mp4"},
        style={"id": "train-01", "prompt": "p"},
        controls=[rt.Control("depth", 0.5)],
        bucket="480",
    )
    assert sample["depth"]["control_path"].endswith("depth/ep000.mp4")


def test_a_declared_but_missing_map_is_refused_rather_than_silently_re_estimated(corpus):
    with pytest.raises(rt.DriverError, match="Refusing"):
        rt.build_sample(
            rt.WorkUnit("u", "ep000", 120, "train-01", 0, 7001),
            source_root=corpus["manifest"].parent,
            episode={"id": "ep000", "frames": 120, "video": "videos/ep000.mp4", "depth": "gone.mp4"},
            style={"id": "train-01", "prompt": "p"},
            controls=[rt.Control("depth", 0.5)],
            bucket="480",
        )
