"""Checks that need no GPU: schema, origin guard, upload path safety."""

import io
import zipfile

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

import app as runner

client = TestClient(runner.app)


def test_defaults_match_schema():
    Draft202012Validator.check_schema(runner.PARAMETER_SCHEMA)
    assert list(Draft202012Validator(runner.PARAMETER_SCHEMA).iter_errors(runner.DEFAULTS)) == []
    assert client.get("/models").json()[0]["default_parameters"] == runner.DEFAULTS


def test_foreign_origin_is_rejected():
    assert client.get("/models", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/models", headers={"Origin": "http://localhost:5173"}).status_code == 200


@pytest.mark.parametrize("name", ["../evil.xml", "/etc/evil.xml", "a/../../evil.xml", "C:\\evil.xml"])
def test_unsafe_upload_paths_are_rejected(tmp_path, name):
    with pytest.raises(HTTPException):
        runner._safe_target(tmp_path, name)


def test_zip_without_scene_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "no scene here")
    response = client.post("/scenes", files={"files": ("scene.zip", buffer.getvalue(), "application/zip")})
    assert response.status_code == 400


def test_run_rejects_bad_parameters():
    response = client.post("/scenes/x/run", json={"parameters": {"rm_max_depth": 99}})
    assert response.status_code == 400
    assert response.json()["detail"][0].startswith("rm_max_depth:")


def test_area_must_have_a_sensible_size():
    response = client.post("/scenes/area", json={"bbox": [9.9, 53.5, 10.0, 53.6]})  # about 6.6 × 11 km
    assert response.status_code == 400 and "each side" in response.json()["detail"]
    assert client.post("/scenes/area", json={"bbox": [9.9, 53.5, 10.0]}).status_code == 400


def test_area_needs_a_cityjson_endpoint(monkeypatch):
    monkeypatch.setattr(runner.lod3_tiles, "SOURCE", "")
    small = [9.99, 53.55, 9.995, 53.553]
    response = client.post("/scenes/area", json={"bbox": small})
    assert response.status_code == 400 and "CityJSON endpoint" in response.json()["detail"]
    assert client.post("/scenes/area", json={"bbox": small, "tile_source": "file:///etc"}).status_code == 400
    assert client.post("/scenes/local", json={"id": "area:centre", "tile_source": "file:///etc"}).status_code == 400


def test_radio_map_covers_only_the_area_of_interest(monkeypatch):
    monkeypatch.setattr(runner, "_bounds", lambda _scene: ([-500.0, -500.0, 0.0], [500.0, 500.0, 30.0]))
    geo = {"origin": [565000, 5934000, 0], "aoi": [564900, 5933950, 565100, 5934050]}
    assert runner._radio_map_area(None, geo) == ([-100, -50, 0.0], [100, 50, 30.0])
    assert runner._radio_map_area(None, {"aoi": None}) == ([-500.0, -500.0, 0.0], [500.0, 500.0, 30.0])


@pytest.mark.parametrize("crs", ["http://www.opengis.net/def/crs/EPSG/0/25832", "urn:ogc:def:crs:EPSG::25832"])
def test_georeference_reads_both_crs_spellings(tmp_path, crs):
    (tmp_path / "scene.json").write_text('{"crs": "%s", "origin": [565000, 5934000, 5]}' % crs)
    geo = runner._georeference(tmp_path / "scene.xml")
    assert geo["crs"] == "EPSG:25832"
    lon, lat = runner._lonlat(geo, 0, 0)
    assert 9.9 < lon < 10.1 and 53.5 < lat < 53.6


def test_local_scene_list_holds_tiles_and_mitsuba_scenes(tmp_path, monkeypatch):
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles" / "a.city.jsonl").write_text("{}")
    (tmp_path / "blender" / "meshes").mkdir(parents=True)
    (tmp_path / "blender" / "town.xml").write_text("<scene/>")
    (tmp_path / "blender" / "scene.json").write_text("{}")  # a georeference, not a scene
    monkeypatch.setattr(runner, "LOCAL_DATA", tmp_path)
    monkeypatch.setattr(runner, "_built_in_scenes", lambda: {"munich": "/sionna/munich.xml"})
    listed = runner._local_scenes()
    assert {key: entry["kind"] for key, entry in listed.items()} == {
        **{key: "area" for key in runner.SAMPLE_AREAS},
        "tiles/a.city.jsonl": "cityjson",
        "blender/town.xml": "mitsuba",
        "builtin:munich": "builtin",
    }


def test_mitsuba_scene_without_sidecar_has_no_georeference(tmp_path):
    assert runner._georeference(tmp_path / "scene.xml") is None


class _Array:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def test_paths_become_3d_lines_strongest_first(monkeypatch):
    import numpy

    monkeypatch.setattr(runner, "np", numpy)
    geo = {
        "origin": [565000, 5934000, 0],
        "to_wgs84": runner.Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True),
    }
    # One rx, one tx, two paths: line of sight, and a reflection then a diffraction.
    kinds = numpy.array([[[[0, 1]]], [[[0, 8]]]])  # [depth, rx, tx, path]
    vertices = numpy.zeros((2, 1, 1, 2, 3))
    vertices[:, 0, 0, 1] = [[5, 5, 10], [8, 2, 4]]
    paths = type("Paths", (), {
        "synthetic_array": True,
        "vertices": _Array(vertices),
        "interactions": _Array(kinds),
        "valid": _Array(numpy.array([[[True, True]]])),
        "tau": _Array(numpy.array([[[1e-7, 2e-7]]])),
        "a": (_Array(numpy.array([[[[[1e-3, 1e-4]]]]])), _Array(numpy.zeros((1, 1, 1, 1, 2)))),
    })()
    devices = [{"position": [0, 0, 20]}], [{"position": [10, 0, 1.5]}]
    features = runner._path_features(paths, geo, *devices)
    assert [feature["properties"]["type"] for feature in features] == ["los", "diffraction"]
    assert features[0]["properties"]["gain_db"] == -60.0
    assert features[1]["properties"]["interactions"] == "reflection → diffraction"
    assert features[1]["properties"]["delay_ns"] == 200.0
    coordinates = features[1]["geometry"]["coordinates"]
    assert len(coordinates) == 4 and [point[2] for point in coordinates] == [20.0, 10.0, 4.0, 1.5]


def test_clearing_storage_deletes_working_files_and_old_start_folders(tmp_path, monkeypatch):
    work = tmp_path / "sionna-runner"
    monkeypatch.setattr(runner, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(runner, "DATA_DIR", work)
    (work / "abc" / "meshes").mkdir(parents=True)
    (work / "abc" / "meshes" / "a.ply").write_bytes(b"12345")
    (tmp_path / "sionna-runner-old1").mkdir()
    (tmp_path / "keep.txt").write_text("not ours")
    assert runner._storage()["bytes"] == 5
    runner._clear_storage()
    assert runner._storage() == {"bytes": 0, "files": 0, "scenes": 0}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt", "sionna-runner"]


def test_a_public_site_may_call_the_runner_on_this_machine():
    # Chrome asks before a public page reaches localhost; the runner says yes, for allowed origins only.
    preflight = {"access-control-request-method": "GET", "access-control-request-private-network": "true"}
    allowed = client.options("/health", headers={"origin": "http://localhost:5173", **preflight})
    assert allowed.headers.get("access-control-allow-private-network") == "true"
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    # Any other site's preflight fails: no allowed origin, so the browser never sends the request.
    other = client.options("/health", headers={"origin": "https://evil.example", **preflight})
    assert other.status_code == 400
    assert "access-control-allow-origin" not in other.headers
