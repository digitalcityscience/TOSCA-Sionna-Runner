import json
import os

from pyproj import Transformer

import app
import lod3_tiles

# Your endpoint from .env (uv run --env-file .env pytest), else an example; nothing is downloaded.
SOURCE = os.environ.get("SIONNA_TILE_SOURCE") or "https://tosca.example.org/sionna/cityjson"


def test_one_cache_folder_per_endpoint():
    assert lod3_tiles.cache("http://tiles.example:8081/catalog/lod3").name == "tiles.example_8081_catalog_lod3"
    assert lod3_tiles.cache("https://a.example/x") != lod3_tiles.cache("https://b.example/x")


def test_tiles_around_an_area(tmp_path, monkeypatch):
    monkeypatch.setattr(lod3_tiles, "CACHE", tmp_path)
    lod3_tiles.cache(SOURCE).mkdir(parents=True)
    # As the catalog's metadata.json: 20 columns from the west, 19 rows from the north.
    (lod3_tiles.cache(SOURCE) / "metadata.json").write_text(json.dumps({"region_extent": [548681, 5917214, 587612, 5953601], "tile_matrix": [20, 19]}))

    # The buildings of the sample tile hamburg_0_7, well inside it.
    assert lod3_tiles.names((549300, 5938400, 550400, 5939900), SOURCE) == ["hamburg_0_7"]
    # At the corner of four tiles.
    assert lod3_tiles.names((550600, 5938250, 550650, 5938300), SOURCE) == ["hamburg_0_7", "hamburg_0_8", "hamburg_1_7", "hamburg_1_8"]
    assert lod3_tiles.names((500000, 5900000, 500100, 5900100), SOURCE) == []


def test_sample_areas_lie_on_the_tiles(tmp_path, monkeypatch):
    monkeypatch.setattr(lod3_tiles, "CACHE", tmp_path)
    lod3_tiles.cache(SOURCE).mkdir(parents=True)
    (lod3_tiles.cache(SOURCE) / "metadata.json").write_text(json.dumps({"region_extent": [548681, 5917214, 587612, 5953601], "tile_matrix": [20, 19]}))
    to_crs = Transformer.from_crs("EPSG:4326", lod3_tiles.CRS, always_xy=True)
    for name, lon, lat, side, buffer in app.SAMPLE_AREAS.values():
        x, y = to_crs.transform(lon, lat)
        assert lod3_tiles.names((x, y, x, y), SOURCE), name
