import json

from geodata_scene import _triangulate, convert

import numpy as np

# A 10 m cube at UTM-sized coordinates, as one CityJSONSeq feature.
HEADER = {
    "type": "CityJSON",
    "version": "2.0",
    "CityObjects": {},
    "vertices": [],
    "transform": {"scale": [0.001, 0.001, 0.001], "translate": [565000.0, 5934000.0, 5.0]},
    "metadata": {"referenceSystem": "http://www.opengis.net/def/crs/EPSG/0/25832"},
}
CUBE = {
    "type": "CityJSONFeature",
    "id": "b1",
    "CityObjects": {
        "b1": {
            "type": "Building",
            "geometry": [
                {
                    "type": "Solid",
                    "lod": "2",
                    "boundaries": [
                        [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]], [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]
                    ],
                    "semantics": {
                        "surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}],
                        "values": [[0, 1, 2, 2, 2, 2]],
                    },
                }
            ],
        }
    },
    "vertices": [
        [0, 0, 0], [10000, 0, 0], [10000, 10000, 0], [0, 10000, 0],
        [0, 0, 10000], [10000, 0, 10000], [10000, 10000, 10000], [0, 10000, 10000],
    ],
}


def test_cube_becomes_a_georeferenced_scene(tmp_path):
    source = tmp_path / "cube.city.jsonl"
    source.write_text(json.dumps(HEADER) + "\n" + json.dumps(CUBE) + "\n", encoding="utf-8")

    result = convert(source, tmp_path / "scene")

    assert result["crs"].endswith("25832")
    # z 5 to 15 in the source: the building is set on the ground at z = 0.
    assert result["origin"] == [565005.0, 5934005.0, 0.0]
    assert result["bounds"]["max"][2] == 10.0
    buildings = json.loads((tmp_path / "scene" / "scene.json").read_text())["buildings"]
    assert buildings == [{"ring": [[565000.0, 5934000.0], [565000.0, 5934010.0], [565010.0, 5934010.0], [565010.0, 5934000.0]], "height": 10.0}]
    # Ground surface dropped; roof 2 and flat ground 2 (both concrete), walls 8.
    assert result["triangles"] == {"itu_concrete": 4, "itu_brick": 8}
    xml = (tmp_path / "scene" / "scene.xml").read_text()
    for material in result["triangles"]:
        assert f"meshes/{material}.ply" in xml and (tmp_path / "scene" / "meshes" / f"{material}.ply").stat().st_size > 0


def test_area_scene_has_ground_over_its_bbox_and_keeps_the_area_of_interest(tmp_path):
    cube, empty = tmp_path / "cube.city.jsonl", tmp_path / "empty.city.jsonl"
    cube.write_text(json.dumps(HEADER) + "\n" + json.dumps(CUBE) + "\n", encoding="utf-8")
    empty.write_text(json.dumps(HEADER) + "\n", encoding="utf-8")
    bbox, aoi = (564900.0, 5933900.0, 565100.0, 5934100.0), (564950.0, 5933950.0, 565050.0, 5934050.0)

    result = convert([cube, empty], tmp_path / "scene", bbox=bbox, aoi=aoi)

    assert result["origin"] == [565000.0, 5934000.0, 0.0]
    assert result["bounds"] == {"min": [564900.0, 5933900.0, 0.0], "max": [565100.0, 5934100.0, 10.0]}
    assert result["aoi"] == list(aoi)
    assert result["triangles"]["itu_concrete"] == 4  # roof 2, ground 2


def test_area_without_buildings_is_ground_only(tmp_path):
    source = tmp_path / "cube.city.jsonl"
    source.write_text(json.dumps(HEADER) + "\n" + json.dumps(CUBE) + "\n", encoding="utf-8")
    result = convert([source], tmp_path / "scene", bbox=(560000.0, 5930000.0, 560100.0, 5930100.0))
    assert result["triangles"] == {"itu_concrete": 2}


def test_hole_is_cut_out():
    outer = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=float)
    hole = np.array([[1, 1, 0], [1, 3, 0], [3, 3, 0], [3, 1, 0]], dtype=float)
    triangles = _triangulate([outer, hole])
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
    assert area == 12.0
