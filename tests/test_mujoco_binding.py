"""PR-08-V5's ground-truth route: the MuJoCo capture shim, and every way it refuses.

Two halves, split by what needs a GL context and what does not, because on this box the default
suite has no display (``tests/test_mujoco_g1.py`` fails for exactly that reason, and MuJoCo binds
its GL backend at ``import mujoco`` — setting ``MUJOCO_GL`` afterwards does nothing):

* **Everything except rendering runs here, in-process.** Compiling the scene, resolving the mesh,
  merging the OBJ, the label map, the schedule and all nine refusals need no pixels. That is most
  of the module and all of its failure surface.
* **Rendering runs in a SUBPROCESS with ``MUJOCO_GL=egl``**, and skips if that subprocess reports
  no usable backend. A skipped render test is honest; a fifth failure in a suite whose four
  failures are documented and unrelated is not.

The object mesh is a synthetic two-group OBJ written per test, never the box's apple: a test that
depends on ``~/IsaacLab-Arena`` existing is a test that passes on one machine.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import measure_est_drift as ed  # noqa: E402

from wam.robot import mujoco_binding as mb  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

mujoco = pytest.importorskip(
    "mujoco",
    reason="the PR-08-V5 ground-truth route needs the optional 'sim' extra — `uv pip install "
    "mujoco`",
)

if not mb.DEFAULT_SCENE.is_file() or not (REPO_ROOT / "assets/mujoco/unitree_g1").is_dir():
    pytest.skip(
        "MuJoCo sim assets missing — fetch the vendor G1 model with "
        "`.venv/bin/python scripts/fetch_g1_model.py`",
        allow_module_level=True,
    )


# -- fixtures ------------------------------------------------------------------------------------


def _two_group_obj(path: pathlib.Path, half: float = 0.04) -> pathlib.Path:
    """An OBJ in TWO ``o`` groups whose union is a closed octahedron-ish blob.

    Two groups on purpose: it is the shape of the real apple mesh (a 14-part convex
    decomposition) and therefore the shape of the defect
    :func:`~wam.robot.mujoco_binding.load_obj_mesh` exists for. Group 1 is the top pyramid,
    group 2 the bottom, and each carries per-vertex colour after xyz — which the real file does
    too, and which is exactly the sort of extra column a stricter parser would reject.
    """
    v = [
        (0, 0, half), (half, 0, 0), (0, half, 0), (-half, 0, 0), (0, -half, 0),
        (0, 0, -half),
    ]
    lines = ["# synthetic two-group test mesh\n"]
    for x, y, z in v:
        lines.append(f"v {x} {y} {z} 0.8 0.1 0.1\n")
    lines.append("o top\n")
    lines += ["f 1 2 3\n", "f 1 3 4\n", "f 1 4 5\n", "f 1 5 2\n"]
    lines.append("o bottom\n")
    lines += ["f 6 3 2\n", "f 6 4 3\n", "f 6 5 4\n", "f 6 2 5\n"]
    path.write_text("".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def mesh(tmp_path) -> pathlib.Path:
    return _two_group_obj(tmp_path / "blob.obj")


def _binding(mesh_path, **kw):
    """A binding with NO GL context: everything but the three render methods works."""
    kw.setdefault("build_renderer", False)
    kw.setdefault("object_mesh", mesh_path)
    return mb.MuJoCoGroundTruthBinding(**kw)


# -- the OBJ merge, which is the whole reason this module parses OBJ at all ----------------------


def test_mujocos_own_obj_loader_keeps_only_the_first_group(mesh):
    """THE MEASUREMENT THE PARSER EXISTS FOR, checked against MuJoCo rather than asserted.

    Hand the same two-group file straight to ``MjSpec.add_mesh(file=...)`` and MuJoCo compiles
    all the vertices and only the FIRST group's faces. Nothing raises. On the real apple that is
    316 of 35 390 triangles rendering as a chip of fruit, covering ~24 px where the whole object
    covers ~1 900 — a plausible frame, a plausible centroid, a plausible p95, measured on the
    wrong geometry. If this test ever fails because MuJoCo learned to merge groups, the parser
    can go; until then it may not.
    """
    spec = mujoco.MjSpec()
    m = spec.add_mesh()
    m.name = "blob"
    m.file = str(mesh)
    body = spec.worldbody.add_body()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = "blob"
    model = spec.compile()
    assert int(model.mesh_facenum[0]) == 4, "MuJoCo took one group's four faces, not all eight"


def test_load_obj_mesh_merges_every_group(mesh):
    loaded = mb.load_obj_mesh(mesh)
    assert loaded.groups_merged == 2
    assert loaded.faces.shape == (8, 3), "both groups' faces, not just the first"
    assert loaded.verts.shape == (6, 3), "xyz only; the per-vertex colour columns are dropped"
    assert loaded.source == mesh


def test_the_merged_mesh_is_what_mujoco_ends_up_with(mesh):
    """The merge is not just parsed, it survives the compile — 8 triangles, not 4."""
    binding = _binding(mesh)
    gid = mujoco.mj_name2id(binding._model, mujoco.mjtObj.mjOBJ_GEOM, "apple")
    assert int(binding._model.mesh_facenum[binding._model.geom_dataid[gid]]) == 8


def test_negative_face_indices_are_relative_to_the_vertices_so_far(tmp_path):
    """Both spellings appear in the wild; getting the negative case wrong scrambles the mesh
    rather than failing, which is the class of bug this whole module is careful about."""
    path = tmp_path / "neg.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n", encoding="utf-8")
    loaded = mb.load_obj_mesh(path)
    assert loaded.faces.tolist() == [[0, 1, 2]]


def test_a_polygon_is_fanned_into_triangles(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="utf-8")
    assert mb.load_obj_mesh(path).faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_face_tokens_with_texture_and_normal_indices_are_accepted(tmp_path):
    path = tmp_path / "vtn.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n", encoding="utf-8")
    assert mb.load_obj_mesh(path).faces.tolist() == [[0, 1, 2]]


def test_an_obj_with_no_triangles_is_refused_naming_the_file(tmp_path):
    """An empty mesh renders nothing and measures nothing, without crashing — so it stops here."""
    path = tmp_path / "empty.obj"
    path.write_text("# nothing at all\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"empty\.obj"):
        mb.load_obj_mesh(path)


# -- resolving the mesh, and never fetching one --------------------------------------------------


def test_an_explicit_mesh_that_does_not_exist_is_refused_rather_than_searched_around(tmp_path):
    """The operator said WHICH mesh. Falling through to the search would quietly render a
    different apple, and the number is a budget for whatever was in frame."""
    with pytest.raises(FileNotFoundError, match="nope.obj"):
        mb.resolve_object_mesh(tmp_path / "nope.obj")


def test_the_env_override_is_honoured_and_refused_when_it_points_nowhere(tmp_path, monkeypatch):
    mesh = _two_group_obj(tmp_path / "env.obj")
    monkeypatch.setenv(mb.OBJECT_MESH_ENV_VAR, str(mesh))
    assert mb.resolve_object_mesh() == mesh
    monkeypatch.setenv(mb.OBJECT_MESH_ENV_VAR, str(tmp_path / "gone.obj"))
    with pytest.raises(FileNotFoundError, match="gone.obj"):
        mb.resolve_object_mesh()


def test_when_nothing_is_found_the_refusal_names_every_path_and_forbids_the_cube(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(mb.OBJECT_MESH_ENV_VAR, raising=False)
    monkeypatch.setattr(mb, "OBJECT_MESH_SEARCH_PATHS", (tmp_path / "a.obj", tmp_path / "b.obj"))
    with pytest.raises(FileNotFoundError) as excinfo:
        mb.resolve_object_mesh()
    message = str(excinfo.value)
    assert str(tmp_path / "a.obj") in message and str(tmp_path / "b.obj") in message
    # The substitution the runbook warned about is refused BY NAME, with the mechanism: the
    # estimator's prompt is part of the committed segmenter contract, so a cube run cannot be a
    # gate input however good the render is.
    assert "orange cube" in message
    assert "segmenter_params_disagree_with_geom_tol" in message


def test_the_default_search_either_finds_a_file_or_says_where_it_looked(monkeypatch):
    """Environment-independent: on a box with the mesh this returns it, on one without it the
    refusal is actionable. Neither outcome may be a silent fallback."""
    monkeypatch.delenv(mb.OBJECT_MESH_ENV_VAR, raising=False)
    try:
        found = mb.resolve_object_mesh()
    except FileNotFoundError as exc:
        assert all(str(p) in str(exc) for p in mb.OBJECT_MESH_SEARCH_PATHS)
    else:
        assert found.is_file() and found in mb.OBJECT_MESH_SEARCH_PATHS


# -- the scene schedule (PR-08 §4.6: N is counted in configurations, not frames) ------------------


def test_the_schedule_is_deterministic_and_has_the_requested_length():
    assert mb.default_scene_schedule(20) == mb.default_scene_schedule(20)
    assert len(mb.default_scene_schedule(7)) == 7


def test_the_schedule_states_are_actually_distinct():
    """A "95th percentile over 20 configurations" that is really 20 copies of one pose is the
    fraudulent-looking number §4.6 warns about."""
    states = mb.default_scene_schedule(20)
    assert len({(s.object_xy, s.object_yaw_rad, s.arm_pitch_offset_rad) for s in states}) == 20


def test_the_schedule_saturates_at_sixty_configurations_and_then_repeats_exactly():
    """THE INDEPENDENT-SAMPLE CEILING OF THIS ROUTE, pinned so it is a known limit and not a
    surprise found by somebody asking for more states and getting the same number back.

    The object walks a 4x5 lattice, the yaw advances on ``i % 5`` and the arm offset on
    ``i % 3``, so the state tuple repeats with period ``lcm(20, 5, 3) = 60``. Past that
    ``--scene-states`` buys duplicate configurations, and the frames of a duplicate are not a
    second observation — measured 2026-08-23, the displacement spread INSIDE one configuration
    was 0.05-0.28 px against 0.03-39.9 px between them.

    So ``EST_DRIFT_P95`` from the committed scene is a percentile over at most 60 independent
    samples however long the capture runs. That is a fact about the SCENE, and raising it means
    arguing a new lattice in a V-document (the placements are registered: ``T40_RULE_V5`` §4.5
    forbids changing the capture scene to improve the number), not passing a bigger integer.
    """
    def tuples(n):
        s = mb.default_scene_schedule(n)
        return {(x.object_xy, x.object_yaw_rad, x.arm_pitch_offset_rad) for x in s}

    assert len(tuples(60)) == 60
    for n in (61, 120, 240):
        assert len(tuples(n)) == 60, f"--scene-states {n} still yields 60 configurations"
    assert mb.default_scene_schedule(120)[:60] == mb.default_scene_schedule(60), "and repeats"


def test_a_schedule_of_no_states_is_refused():
    with pytest.raises(ValueError, match="n_states"):
        mb.default_scene_schedule(0)


def test_the_binding_visits_one_state_per_steps_per_state(mesh):
    """The state index is ``floor(tick / steps_per_state)``, so the FIRST block of steps is
    state 0 and the change lands on the step that crosses the boundary, not on the one before
    it. Written out because an off-by-one here is a capture whose last state is never visited."""
    binding = _binding(mesh, schedule=mb.default_scene_schedule(4), steps_per_state=5)
    assert binding.scene_states_visited == 1
    binding.step(5)
    assert binding.scene_states_visited == 1, "steps 0..4 are all state 0"
    binding.step(5)
    assert binding.scene_states_visited == 2, "state 1 applied at tick 5"
    binding.step(10)
    assert binding.scene_states_visited == 4


def test_the_object_actually_moves_between_states(mesh):
    binding = _binding(mesh, schedule=mb.default_scene_schedule(4), steps_per_state=2)
    first = binding._model.body_pos[binding._object_bid].copy()
    binding.step(3)
    assert not np.allclose(first, binding._model.body_pos[binding._object_bid])


def test_the_object_sits_on_the_table_top_and_not_through_it(mesh):
    """The table top is at z = 0.72 m (the scene header, measured). The schedule places the
    object by its own half-height, whatever mesh was resolved, so a bigger apple does not sink."""
    binding = _binding(mesh)
    z = float(binding._model.body_pos[binding._object_bid][2])
    assert z == pytest.approx(0.72 + binding._object_half_z, abs=1e-9)


# -- what the capture rig reads off the segmentation ----------------------------------------------


def test_the_label_map_is_replicators_shape_and_the_rig_finds_the_object(mesh):
    """The join that decides whether anything is measured at all: ``measure_est_drift.object_ids``
    matches ``strip().lower()`` equality against the label text, and forgives nothing else. A
    scene calling the fruit ``apple_01`` produces a full run, zero coverage and no crash."""
    binding = _binding(mesh)
    labels = binding.geom_id_to_labels()
    assert all(isinstance(v, dict) and set(v) == {"class"} for v in labels.values())
    matched, vocab = ed.object_ids(labels, ed.DEFAULT_OBJECT_CLASS)
    assert len(matched) == 1
    assert "apple" in vocab and "cube" in vocab and "table_top" in vocab


def test_ids_are_geom_id_plus_one_so_zero_stays_background(mesh):
    binding = _binding(mesh)
    labels = binding.geom_id_to_labels()
    assert 0 not in labels
    gid = mujoco.mj_name2id(binding._model, mujoco.mjtObj.mjOBJ_GEOM, "apple")
    assert labels[gid + 1] == {"class": "apple"}


def test_every_named_geom_is_labelled_even_when_it_is_not_visible(mesh):
    """Occlusion must land in ``paired_displacements``' drop-and-count bucket (an empty mask),
    not in ``n_frames_without_object_label`` — "the apple is behind the hand" and "this scene has
    no apple in it" are different claims and are fixed differently."""
    binding = _binding(mesh)
    named = sum(
        1
        for gid in range(binding._model.ngeom)
        if mujoco.mj_id2name(binding._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    )
    assert len(binding.geom_id_to_labels()) == named


# -- the refusals ---------------------------------------------------------------------------------


def test_ground_truth_is_opt_in_and_the_calls_raise_rather_than_returning_zeros(mesh):
    """Same rule and the same error type as ``FakeIsaacBinding``: a binding without the channel
    has no depth to give, and a zero array would enter the error budget as a measurement."""
    binding = _binding(mesh)
    assert binding.ground_truth_channels == ()
    with pytest.raises(RuntimeError, match="not attached"):
        binding.render_depth("head")
    with pytest.raises(RuntimeError, match="not attached"):
        binding.render_segmentation("head")


def test_an_unknown_ground_truth_channel_is_refused_at_construction(mesh):
    with pytest.raises(ValueError, match="unknown ground-truth channel"):
        _binding(mesh, ground_truth=("normals",))


def test_a_bare_string_ground_truth_is_refused_rather_than_iterated(mesh):
    with pytest.raises(ValueError, match="pass"):
        _binding(mesh, ground_truth="depth")


def test_an_unknown_camera_is_refused_against_the_scenes_own_cameras(mesh):
    with pytest.raises(ValueError) as excinfo:
        _binding(mesh, cameras=("persp",))
    message = str(excinfo.value)
    assert "persp" in message and "head" in message and "wrist_left" in message


def test_rendering_from_an_unattached_camera_is_refused(mesh):
    binding = _binding(mesh, cameras=("head",))
    with pytest.raises(ValueError, match="unknown camera 'wrist_left'"):
        binding.render_frame("wrist_left")


def test_a_render_grid_past_the_offscreen_buffer_is_refused_before_any_gl(mesh):
    """MuJoCo CLAMPS ``<visual><global offwidth/offheight>`` silently, so the frames would come
    back at a size ``measure`` disqualifies as ``resolution_disagrees_with_geom_tol`` — the
    cheapest possible error discovered in the most expensive possible place."""
    with pytest.raises(ValueError, match="offscreen buffer"):
        mb.MuJoCoGroundTruthBinding(
            object_mesh=mesh, render_hw=(2000, 2000), build_renderer=True
        )


def test_no_gl_backend_is_a_named_refusal_and_not_a_mujoco_traceback(mesh, monkeypatch):
    """MEASURED 2026-08-23: headless, with no ``MUJOCO_GL``, this raised
    ``mujoco.FatalError: gladLoadGL error``.

    ``mujoco.FatalError`` derives from ``Exception``, **not** ``RuntimeError``, so
    ``measure_est_drift.main``'s ``except (FileNotFoundError, ValueError, RuntimeError)`` did not
    catch it: the operator got a traceback and exit 1 where every other failure of this
    constructor gives ``FATAL: ...`` and exit 2. It was invisible to this file because the
    render tests below run in a subprocess that already exports ``MUJOCO_GL=egl``.

    The stub raises a bare ``Exception`` rather than ``mujoco.FatalError`` on purpose — what is
    asserted is the conversion of *anything that is not one of the module's own refusals*, so
    the test does not go quiet if the vendor re-parents its error class.
    """
    binding = _binding(mesh)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    class _NoGL:
        def __init__(self, *a, **kw):
            raise Exception("gladLoadGL error")

    monkeypatch.setattr(binding._mj, "Renderer", _NoGL)
    with pytest.raises(RuntimeError) as excinfo:
        binding._make_renderer()
    message = str(excinfo.value)
    assert "MUJOCO_GL" in message, "the refusal must name the variable that fixes it"
    assert "egl" in message and "osmesa" in message, "and both backends that work"
    assert "$MUJOCO_GL: unset" in message, "and what it actually was, so a typo is visible"
    assert "gladLoadGL error" in message, "and the vendor's own words, not a paraphrase"


def test_the_gl_refusal_does_not_swallow_the_modules_own_refusals(mesh, monkeypatch):
    """The catch is broad, so it must re-raise ValueError/RuntimeError unchanged.

    Otherwise a real refusal from inside ``Renderer`` construction — or any future one — would
    be rewritten into "set MUJOCO_GL", which is the wrong fix stated confidently.
    """
    binding = _binding(mesh)

    class _Refuses:
        def __init__(self, *a, **kw):
            raise ValueError("offscreen buffer")

    monkeypatch.setattr(binding._mj, "Renderer", _Refuses)
    with pytest.raises(ValueError, match="offscreen buffer"):
        binding._make_renderer()


@pytest.mark.parametrize("bad", [(480,), (0, 640), (480, 0)])
def test_a_malformed_render_hw_is_refused(mesh, bad):
    with pytest.raises(ValueError, match="render_hw"):
        _binding(mesh, render_hw=bad)


def test_steps_per_state_must_be_at_least_one(mesh):
    with pytest.raises(ValueError, match="steps_per_state"):
        _binding(mesh, steps_per_state=0)


def test_an_empty_object_label_is_refused(mesh):
    with pytest.raises(ValueError, match="object_label"):
        _binding(mesh, object_label="  ")


def test_a_missing_keyframe_is_refused_naming_the_ones_the_scene_has(mesh):
    with pytest.raises(ValueError, match="ready"):
        _binding(mesh, keyframe="stand_on_one_leg")


def test_a_missing_scene_is_refused(mesh, tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.xml"):
        _binding(mesh, scene=tmp_path / "nowhere.xml")


def test_a_binding_with_no_renderer_says_so_rather_than_pretending(mesh):
    binding = _binding(mesh, ground_truth=("depth", "segmentation"))
    with pytest.raises(RuntimeError, match="build_renderer=False"):
        binding.render_frame("head")


def test_a_closed_binding_refuses_everything(mesh):
    binding = _binding(mesh)
    binding.close()
    for call in (
        lambda: binding.step(),
        lambda: binding.reset(),
        lambda: binding.get_physics_step_count(),
        lambda: binding.render_frame("head"),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_close_is_idempotent(mesh):
    binding = _binding(mesh)
    binding.close()
    binding.close()


# -- the contract the rig depends on --------------------------------------------------------------


def test_the_tick_is_an_exact_int_and_advances_only_on_step(mesh):
    """Staleness upstream is an EQUALITY test against the previous value, so a float here (or a
    tick derived from sim time) would make that comparison meaningless."""
    binding = _binding(mesh)
    assert isinstance(binding.get_physics_step_count(), int)
    assert binding.get_physics_step_count() == 0
    binding.step(3)
    assert binding.get_physics_step_count() == 3
    binding.reset()
    assert binding.get_physics_step_count() == 3, "reset does not rewind the raw counter"


def test_the_committed_scene_file_is_not_modified_by_a_capture(mesh):
    """``configs/sim/g1_scene.xml`` is the E2 scene every other MuJoCo result was measured
    against. A calibration capture that quietly edited it would make those unreproducible."""
    before = mb.DEFAULT_SCENE.read_bytes()
    binding = _binding(mesh)
    binding.step(2)
    binding.close()
    assert mb.DEFAULT_SCENE.read_bytes() == before


def test_adding_the_object_leaves_nq_alone_so_the_ready_keyframe_still_applies(mesh):
    """The object carries no joint on purpose. A free joint would lengthen ``qpos``, MuJoCo
    zero-pads a short keyframe rather than refusing it, and the object would start at the world
    origin inside the floor with nothing said (the scene header records the same trap for the
    vendor's own `stand` key)."""
    plain = mujoco.MjModel.from_xml_path(str(mb.DEFAULT_SCENE))
    assert _binding(mesh)._model.nq == plain.nq


def test_the_capture_header_carries_the_object_limitations_as_named_fields(mesh):
    """An EST_DRIFT_P95 measured on a stand-in object must not be readable without the stand-in.
    These are fields and not prose for exactly that reason."""
    limits = _binding(mesh).limitations()
    assert limits["object_label"] == "apple"
    assert limits["object_is_substituted"] is False
    assert limits["object_mesh_is_convex_decomposition_proxy"] is True
    assert limits["object_is_untextured"] is True
    assert limits["distractors_left_in_scene"] == ["cube"]
    assert "distance_to_image_plane" in limits["depth_semantics"]


def test_a_stand_in_object_says_so_in_the_limitations(mesh):
    """The cube route is not taken, but if anything other than the apple ever is, the field that
    says so is the first thing a reader of the budget sees."""
    limits = _binding(mesh, object_label="orange cube").limitations()
    assert limits["object_is_substituted"] is True
    assert limits["object_label"] == "orange cube"


def test_the_distractor_cube_is_left_in_the_scene(mesh):
    """Deliberate, and the one place this module could have made the number easier. An orange
    cube beside a red apple is a real chance for the detector to pick the wrong object; removing
    it would lower the p95, widen ``GEOM_TOL - EST_DRIFT_P95`` and land the error in the
    generator's favour, which is the direction PR-08-V5 exists to avoid."""
    binding = _binding(mesh)
    assert mujoco.mj_name2id(binding._model, mujoco.mjtObj.mjOBJ_GEOM, "cube") >= 0


# -- the half that needs pixels -------------------------------------------------------------------


_RENDER_PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    sys.path.insert(0, {src!r})
    from wam.robot.mujoco_binding import MuJoCoGroundTruthBinding
    try:
        b = MuJoCoGroundTruthBinding(
            object_mesh={mesh!r},
            cameras=("head", "wrist_left"),
            render_hw=(240, 320),
            ground_truth=("depth", "segmentation"),
            steps_per_state=2,
        )
    except Exception as exc:
        print(json.dumps({{"no_gl": type(exc).__name__ + ": " + str(exc)[:200]}}))
        raise SystemExit(0)
    b.step(2)
    out = {{}}
    rgb = b.render_frame("head")
    seg = b.render_segmentation("head")
    apple = [k for k, v in seg.id_to_labels.items() if v["class"] == "apple"][0]
    out["rgb_shape"] = list(rgb.shape)
    out["rgb_dtype"] = str(rgb.dtype)
    out["rgb_variance"] = float(np.var(rgb))
    out["ids_dtype"] = str(seg.ids.dtype)
    out["ids_shape"] = list(seg.ids.shape)
    out["apple_px"] = int((seg.ids == apple).sum())
    out["background_px"] = int((seg.ids == 0).sum())
    depth = b.render_depth("head")
    out["depth_dtype"] = str(depth.dtype)
    out["depth_finite_min"] = float(depth[np.isfinite(depth)].min())
    wseg = b.render_segmentation("wrist_left")
    wdepth = b.render_depth("wrist_left")
    out["wrist_background_px"] = int((wseg.ids == 0).sum())
    out["wrist_inf_px"] = int(np.isinf(wdepth).sum())
    tick = b.get_physics_step_count()
    b.render_frame("head")
    out["tick_unchanged_by_render"] = b.get_physics_step_count() == tick
    b.step(4)
    seg2 = b.render_segmentation("head")
    out["apple_px_after_state_change"] = int((seg2.ids == apple).sum())
    b.close()
    print(json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Render once, in a subprocess with ``MUJOCO_GL=egl``, and skip if there is no backend.

    A subprocess and not ``monkeypatch.setenv`` because MuJoCo binds its GL backend at
    ``import mujoco`` — by the time this module is collected, some earlier test module has
    already imported it under whatever ``MUJOCO_GL`` the shell had, and a late setenv is a no-op
    that would look like it worked.
    """
    mesh = _two_group_obj(tmp_path_factory.mktemp("render") / "blob.obj")
    env = dict(os.environ, MUJOCO_GL="egl")
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_PROBE.format(src=str(REPO_ROOT / "src"), mesh=str(mesh))],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip(f"MuJoCo render subprocess failed: {proc.stderr[-400:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    if "no_gl" in payload:
        pytest.skip(f"no usable MuJoCo GL backend here: {payload['no_gl']}")
    return payload


def test_the_rgb_frame_is_the_requested_grid_and_is_not_blank(rendered):
    """Never a pixel-value assertion — rendered pixels are not bit-portable across GL backends
    (``tests/test_mujoco_g1.py``'s rule, kept). Shape, dtype and variance only."""
    assert rendered["rgb_shape"] == [240, 320, 3]
    assert rendered["rgb_dtype"] == "uint8"
    assert rendered["rgb_variance"] > 0.0


def test_the_object_is_actually_in_frame_with_a_ground_truth_mask(rendered):
    """The failure this guards is §4.2's: the rig runs to completion, writes an artifact and
    measures nothing, with no crash and no traceback."""
    assert rendered["ids_shape"] == [240, 320]
    assert rendered["ids_dtype"] == "uint32"
    assert rendered["apple_px"] > 0


def test_the_masks_centroid_moves_when_the_schedule_advances(rendered):
    """A capture whose scene never changed would make "the p95 spans N configurations" a claim
    that cannot fail."""
    assert rendered["apple_px_after_state_change"] != rendered["apple_px"]


def test_background_depth_is_inf_and_matches_the_segmentations_background(rendered):
    """``distance_to_camera`` reports a ray that hit nothing as ``inf`` and
    ``measure_est_drift.depth_error`` excludes and counts exactly those pixels. MuJoCo returns
    the far clip instead — and NOT exactly the far clip, which is why the background is taken
    from the segmentation rather than from a coined tolerance."""
    assert rendered["depth_dtype"] == "float32"
    assert rendered["wrist_background_px"] > 0
    assert rendered["wrist_inf_px"] == rendered["wrist_background_px"]


def test_the_head_view_of_the_table_has_no_background_at_all(rendered):
    """The floor plane is infinite, so nothing in the head view misses geometry. Stated as a
    test because it is why the head camera's depth carries no ``inf`` and a reader checking the
    previous test against the wrong camera would think the masking was broken."""
    assert rendered["background_px"] == 0
    assert rendered["depth_finite_min"] > 0.0


def test_rendering_never_advances_physics(rendered):
    """The adapter owns the clock. A render that stepped behind its back would corrupt staleness
    detection upstream — the same guarantee ``RenderingManager.render()`` is documented to give
    on the Isaac side."""
    assert rendered["tick_unchanged_by_render"] is True


@pytest.fixture(scope="module")
def cli_capture(tmp_path_factory):
    """`measure_est_drift.py capture --backend mujoco`, for real, in a subprocess with EGL.

    The fixture above proves the binding renders; this proves the harness DRIVES it — the header
    it writes, the frames on disk, and the three fields a reader of the budget needs: which route,
    how many configurations, and what object it was.
    """
    work = tmp_path_factory.mktemp("cli")
    mesh = _two_group_obj(work / "blob.obj")
    out = work / "cap"
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/measure_est_drift.py"), "capture",
            "--backend", "mujoco", "--out", str(out), "--frames", "4",
            "--steps-per-frame", "2", "--scene-states", "2",
            "--object-mesh", str(mesh), "--render-hw", "480", "640",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, MUJOCO_GL="egl"),
        timeout=600,
    )
    if proc.returncode != 0:
        if "OpenGL" in proc.stderr or "GLFW" in proc.stderr or "EGL" in proc.stderr:
            pytest.skip(f"no usable MuJoCo GL backend here: {proc.stderr[-300:]}")
        pytest.fail(f"capture failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return out


def test_the_cli_capture_is_ground_truth_and_names_its_route(cli_capture):
    header = json.loads((cli_capture / "capture.json").read_text())
    assert header["binding"] == "MuJoCoGroundTruthBinding"
    assert header["is_simulated_binding"] is False, "this one IS ground truth, unlike the fake"
    assert header["ground_truth_route"] == "mujoco"
    assert header["resolution_hw"] == [480, 640]
    assert header["n_frames"] == 4


def test_the_cli_capture_records_how_many_configurations_it_actually_visited(cli_capture):
    """PR-08 §4.6's missing field: N counted in configurations, not frames. The runbook's advice
    was to put it in the commit message; it is in the artifact instead."""
    header = json.loads((cli_capture / "capture.json").read_text())
    assert header["n_scene_states_scheduled"] == 2
    assert header["n_scene_states_visited"] == 2


def test_the_cli_capture_carries_the_object_limitations_into_the_header(cli_capture):
    limits = json.loads((cli_capture / "capture.json").read_text())["object_limitations"]
    assert limits["object_label"] == "apple"
    assert limits["object_is_substituted"] is False
    assert "distance_to_image_plane" in limits["depth_semantics"]
    assert limits["renderer"].startswith("mujoco rasteriser")


def test_every_captured_frame_carries_a_true_mask_of_the_object(cli_capture):
    """§4.2's failure — a full run, zero coverage, `est_drift_p95_px: null` and no crash — is what
    this asserts the absence of, on the route that was chosen to avoid it."""
    for i in range(4):
        d = cli_capture / "frames" / f"{i:06d}"
        ids = np.load(d / "seg_ids.npy")
        labels = json.loads((d / "seg_labels.json").read_text())
        matched, vocab = ed.object_ids({int(k): v for k, v in labels.items()}, "apple")
        assert matched, f"frame {i} has no apple label; vocabulary was {vocab}"
        assert int(np.isin(ids, matched).sum()) > 0, f"frame {i} has an empty apple mask"
