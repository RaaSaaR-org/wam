"""The review page carries verdict controls; what it must never do is invent a tile.

The load-bearing risk in this builder is silent: a crop taken at the wrong offset shows a reviewer
frame B while the page records their judgement against frame A. Nothing downstream can detect that,
so the geometry is derived from the sheet's own pixels and the builder refuses when the arithmetic
does not close. These tests pin that refusal, and pin that the two sections keep the two different
vocabularies their two different questions need.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module():
    spec = importlib.util.spec_from_file_location(
        "_brp", REPO_ROOT / "scripts" / "build_review_page.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


brp = _module()
Image = pytest.importorskip("PIL.Image", reason="Pillow is needed to crop tiles")


def make_sheet(path: pathlib.Path, n_tiles: int, tile_w: int = 320, tile_h: int = 300) -> None:
    """A sheet laid out exactly as ``audit_apple_masks.contact_sheet`` lays one out."""
    from PIL import Image as PILImage

    cols = brp.SHEET_COLS
    rows = (n_tiles + cols - 1) // cols
    width = cols * tile_w + brp.SHEET_GAP * (cols + 1)
    height = brp.SHEET_HEADER + rows * tile_h + brp.SHEET_GAP * (rows + 1)
    sheet = PILImage.new("RGB", (width, height), (10, 10, 12))
    for i in range(n_tiles):
        row, col = divmod(i, cols)
        tile = PILImage.new("RGB", (tile_w, tile_h), (i * 17 % 256, 40, 200))
        sheet.paste(
            tile,
            (
                brp.SHEET_GAP + col * (tile_w + brp.SHEET_GAP),
                brp.SHEET_HEADER + brp.SHEET_GAP + row * (tile_h + brp.SHEET_GAP),
            ),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def test_geometry_inverts_contact_sheets_own_layout():
    for n_tiles, tile_h in ((12, 300), (12, 287), (6, 300), (2, 271)):
        cols = brp.SHEET_COLS
        rows = (n_tiles + cols - 1) // cols
        width = cols * 320 + brp.SHEET_GAP * (cols + 1)
        height = brp.SHEET_HEADER + rows * tile_h + brp.SHEET_GAP * (rows + 1)
        assert brp.tile_geometry((width, height), n_tiles) == (320, tile_h, rows)


def test_a_sheet_whose_arithmetic_does_not_close_is_refused():
    """One pixel off is a crop at an offset, which is a verdict attached to the wrong frame."""
    with pytest.raises(brp.BuildError) as excinfo:
        brp.tile_geometry((1311, 952), 12)
    assert "does not divide" in str(excinfo.value)


def test_crops_land_on_the_tiles_and_not_between_them(tmp_path):
    """Each crop must be one whole tile: a single flat colour, no gap pixels, no header."""
    from PIL import Image as PILImage

    path = tmp_path / "sheets" / "grasp-00.png"
    make_sheet(path, 12)
    encoded = brp.crop_tiles(path, 12)
    assert len(encoded) == 12
    import base64

    for i, blob in enumerate(encoded):
        tile = PILImage.open(io.BytesIO(base64.b64decode(blob))).convert("RGB")
        assert tile.size == (320, 300)
        expected = (i * 17 % 256, 40, 200)
        for corner in ((2, 2), (317, 2), (2, 297), (317, 297)):
            for got, want in zip(tile.getpixel(corner), expected):
                assert abs(got - want) <= 6, f"tile {i} corner {corner} is not the tile's own colour"


def test_a_short_last_sheet_is_cropped_at_its_own_row_count(tmp_path):
    path = tmp_path / "sheets" / "occluded-00.png"
    make_sheet(path, 6)
    assert len(brp.crop_tiles(path, 6)) == 6


def test_the_two_sections_do_not_share_a_vocabulary():
    """The apple question and the tail question are different questions."""
    assert set(brp.MASK_VERDICTS) & set(brp.TAIL_VERDICTS) == {"undecidable"}
    assert "apple" not in brp.TAIL_VERDICTS
    assert "table" not in brp.MASK_VERDICTS
    for verdict in brp.MASK_VERDICTS + brp.TAIL_VERDICTS:
        assert verdict in brp.VERDICT_LABELS


def test_a_tile_offers_only_its_own_sections_verdicts():
    tile = {
        "section": "tail",
        "sheet": "area-tail-00",
        "key": "episode_000004:221",
        "episode": "episode_000004",
        "frame": 221,
        "group": "tail",
        "note": "",
        "flags": ["mismatch"],
        "jpeg": "",
    }
    rendered = brp.render_tile(tile, 1)
    for verdict in brp.TAIL_VERDICTS:
        assert f'data-verdict="{verdict}"' in rendered
    assert 'data-verdict="apple"' not in rendered
    assert 'data-key="episode_000004:221"' in rendered
    assert "mismatch" in rendered


def test_the_page_carries_no_verdict_and_names_no_bound():
    """It is a recording instrument. A prefilled verdict would be the correlated observer."""
    tiles = [
        {
            "section": "mask",
            "sheet": "census-00",
            "key": "episode_000000:0",
            "episode": "episode_000000",
            "frame": 0,
            "group": "census",
            "note": "",
            "flags": [],
            "jpeg": "",
        },
        {
            "section": "tail",
            "sheet": "area-tail-00",
            "key": "episode_000004:221",
            "episode": "episode_000004",
            "frame": 221,
            "group": "tail",
            "note": "",
            "flags": [],
            "jpeg": "",
        },
    ]
    page = brp.render_page(tiles, {"built_utc": "2026-08-26T00:00:00+00:00", "git_commit": "0" * 40})
    start = page.index('<script type="application/json" id="wam-state">')
    state = json.loads(page[page.index(">", start) + 1 : page.index("</script>", start)])
    assert state["verdicts"] == {}
    assert state["reviewer"] == ""
    assert "max_frame_fraction" not in page
    # No TILE ships pre-pressed; the only aria-pressed in the page is the stylesheet's selector.
    assert "aria-pressed" not in brp.render_tile(tiles[0], 1)
    assert "aria-pressed" not in brp.render_tile(tiles[1], 2)


def test_the_sheet_mapping_is_the_ingest_tools_own(tmp_path):
    """Two implementations of frame->sheet is how a verdict gets attached to an unseen frame."""
    frames = [{"episode": "episode_000000", "frame_index": i, "stratum": "grasp"} for i in range(13)]
    index = brp.load_sheet_index({"frames": frames})
    assert index[0] == "grasp-00"
    assert index[11] == "grasp-00"
    assert index[12] == "grasp-01"
