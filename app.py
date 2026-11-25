"""TOSCA-Sionna Runner: Sionna RT on this machine's GPU for the TOSCA radio planning panel.

Open an area of CityJSON buildings (or upload a scene), set the same parameters
as the Sionna RT desktop GUI and run them on this machine's GPU. ``GET /models``
returns the parameter schema and defaults (key, version, parameter_schema,
default_parameters).
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator
from pyproj import Transformer

import lod3_tiles
from geodata_scene import convert

VERSION = "0.1.0"
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get(
        "SIONNA_RUNNER_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
}
TOKEN = os.environ.get("SIONNA_RUNNER_TOKEN", "")
SCENE_SUFFIXES = {".xml", ".ply", ".obj", ".exr", ".png", ".jpg", ".jpeg", ".json", ".jsonl"}
MAX_RADIO_MAP_CELLS = 4_000_000  # keeps one radio map inside GPU memory
MAX_PATHS = 10_000  # what the map still draws smoothly; the strongest are kept
MAX_AREA_SIDE = 3000.0  # metres; a larger area of interest is too many buildings and radio map cells for one run
AREA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["bbox"],
    "properties": {
        # West, south, east, north in degrees (WGS84).
        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "buffer_m": {"type": "number", "minimum": 0, "maximum": 1000, "default": 300},
        # The TOSCA site's Sionna CityJSON endpoint; SIONNA_TILE_SOURCE when not sent.
        "tile_source": {"type": "string", "pattern": "^https?://"},
    },
}
# Ready-made areas in the panel's scene list, opened like a drawn area:
# id -> (name, centre lon, lat, side of the square area in metres, buffer in metres).
SAMPLE_AREAS = {
    "area:centre": ("Hamburg centre: Rathaus, Binnenalster", 9.9925, 53.5505, 600, 200),
    "area:west": ("Hamburg west: Altona-Ottensen", 9.9300, 53.5530, 600, 200),
    "area:east": ("Hamburg east: Bergedorf", 10.2120, 53.4880, 600, 200),
    "area:rural": ("Hamburg rural: Kirchwerder, Vier- und Marschlande", 10.1990, 53.4210, 1200, 200),
}
# Sionna's InteractionType bits, named for the map.
INTERACTIONS = {1: "reflection", 2: "scattering", 4: "refraction", 8: "diffraction"}
# Uploaded and converted scenes. Only the open scene stays here; the folder is emptied
# when the runner starts and stops. Results are never written: they go to the browser.
# shortcut: one fixed folder, so two runners on one machine would clear each other's files.
WORK_ROOT = Path(os.environ.get("SIONNA_WORK_DIR") or tempfile.gettempdir())
DATA_DIR = WORK_ROOT / "sionna-runner"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# Sample data offered in the panel's scene list: Mitsuba scenes (.xml) and CityJSON tiles.
LOCAL_DATA = Path(os.environ.get("SIONNA_DATA_DIR") or Path(__file__).resolve().parent / "data")

# -- parameters (names and defaults follow sionna-rt-gui's config.py) -----------

FEATURES = {
    "los": "Line of sight",
    "specular_reflection": "Specular reflection",
    "diffuse_reflection": "Diffuse reflection",
    "refraction": "Refraction",
    "diffraction": "Diffraction",
    "edge_diffraction": "Edge diffraction",
    "diffraction_lit_region": "Diffraction lit region",
}


def _number(title: str, default: float, minimum: float, maximum: float) -> dict:
    return {"type": "number", "title": title, "default": default, "minimum": minimum, "maximum": maximum}


def _integer(title: str, default: int, minimum: int, maximum: int) -> dict:
    return {"type": "integer", "title": title, "default": default, "minimum": minimum, "maximum": maximum}


def _choice(title: str, default: str, options: list[str]) -> dict:
    return {"enum": options, "title": title, "default": default}


def _flag(title: str, default: bool) -> dict:
    return {"type": "boolean", "title": title, "default": default}


def _array_fields(prefix: str) -> dict:
    return {
        f"{prefix}_pattern": _choice("Pattern", "iso", ["iso", "dipole", "hw_dipole", "tr38901"]),
        f"{prefix}_polarization": _choice("Polarization", "V", ["V", "H", "VH", "cross"]),
        f"{prefix}_num_rows": _integer("Rows", 1, 1, 64),
        f"{prefix}_num_cols": _integer("Columns", 1, 1, 64),
        f"{prefix}_vertical_spacing": _number("Vertical spacing", 0.5, 0.01, 10),
        f"{prefix}_horizontal_spacing": _number("Horizontal spacing", 0.5, 0.01, 10),
    }


def _feature_fields(prefix: str, enabled: set[str]) -> dict:
    return {f"{prefix}_{name}": _flag(title, name in enabled) for name, title in FEATURES.items()}


GROUPS: dict[str, dict] = {
    "Scene": {
        # Not in the desktop GUI (fixed 3.5 GHz there); needed for n78 / FR3 / mmWave.
        "frequency_ghz": _number("Frequency (GHz)", 3.5, 0.1, 300),
        "view": _choice("Preview camera", "oblique", ["oblique", "top"]),
    },
    "TX array": _array_fields("tx"),
    "RX array": _array_fields("rx"),
    "Radio map": {
        "rm_cell_size_x": _number("Cell size X (m)", 1.0, 0.01, 100),
        "rm_cell_size_y": _number("Cell size Y (m)", 1.0, 0.01, 100),
        "rm_height_above_ground": _number("Height above ground (m)", 1.5, 0, 1000),
        "rm_log_samples": _number("Samples (log 10)", 8.0, 0, 9),
        "rm_max_depth": _integer("Max depth", 5, 1, 10),
        **_feature_fields(
            "rm", {"los", "specular_reflection", "diffuse_reflection", "refraction", "diffraction_lit_region"}
        ),
        "rm_color_map": _choice("Colormap", "viridis", ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]),
        "rm_vmin": _number("vmin (dB)", -150, -200, 0),
        "rm_vmax": _number("vmax (dB)", -50, -200, 0),
    },
    "Paths": {
        "paths_max_depth": _integer("Max depth", 5, 1, 10),
        "paths_synthetic_array": _flag("Synthetic array", True),
        **_feature_fields("paths", {"los", "specular_reflection"}),
    },
}
PROPERTIES = {name: field for fields in GROUPS.values() for name, field in fields.items()}
DEFAULTS = {name: field["default"] for name, field in PROPERTIES.items()}
PARAMETER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": PROPERTIES,
    "additionalProperties": False,
    "x-order": list(PROPERTIES),
    "x-groups": {title: list(fields) for title, fields in GROUPS.items()},
}
_POSITION = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_DEVICES = {
    "type": "array",
    "maxItems": 64,
    "items": {
        "type": "object",
        "required": ["position"],
        "additionalProperties": False,
        "properties": {
            "position": _POSITION,
            # Sionna's alpha (around z) and beta (around y), in degrees.
            "yaw_deg": {"type": "number"},
            "pitch_deg": {"type": "number"},
        },
    },
}
RUN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "parameters": PARAMETER_SCHEMA,
        "transmitters": _DEVICES,
        # A receiver route is one receiver per route point (the web panel sends up to 100 per route).
        "receivers": {**_DEVICES, "maxItems": 512},
        "radio_map": {"type": "boolean"},
        "paths": {"type": "boolean"},
    },
}

# -- Sionna (everything below runs on the one GPU thread) -----------------------

rt = mi = np = plt = None
SCENES: dict[str, object] = {}
# Per scene: where its (0, 0, 0) is on the map. Only scenes made from geodata have one.
GEO: dict[str, dict] = {}


def _georeference(xml_path: Path) -> dict | None:
    """The ``scene.json`` that ``geodata_scene.convert`` writes next to the XML."""
    sidecar = xml_path.parent / "scene.json"
    code = None
    if sidecar.is_file():
        described = json.loads(sidecar.read_text(encoding="utf-8"))
        # "http://www.opengis.net/def/crs/EPSG/0/25832" or "urn:ogc:def:crs:EPSG::25832"
        code = re.search(r"EPSG\S*?[/:](\d+)$", str(described.get("crs", "")))
    if code is None:
        return None
    epsg = f"EPSG:{code.group(1)}"
    return {
        "crs": epsg,
        "origin": described["origin"],
        "buildings": described.get("buildings", []),
        "aoi": described.get("aoi"),
        "to_wgs84": Transformer.from_crs(epsg, "EPSG:4326", always_xy=True),
        "from_wgs84": Transformer.from_crs("EPSG:4326", epsg, always_xy=True),
    }


def _lonlat(geo: dict, x: float, y: float) -> list[float]:
    return list(geo["to_wgs84"].transform(geo["origin"][0] + x, geo["origin"][1] + y))


def _buildings(geo: dict) -> dict:
    """Footprints with heights as GeoJSON, for the map's extrusion layer."""
    features = []
    for building in geo["buildings"]:
        ring = [list(geo["to_wgs84"].transform(x, y)) for x, y in building["ring"]]
        features.append(
            {
                "type": "Feature",
                "properties": {"height": building["height"]},
                "geometry": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _png(rgba) -> str:
    buffer = io.BytesIO()
    plt.imsave(buffer, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _radio_map_png(gain, parameters: dict) -> str:
    """The radio map as a north-up image: transparent where nothing arrives."""
    with np.errstate(divide="ignore"):
        decibels = 10 * np.log10(gain)
    span = max(parameters["rm_vmax"] - parameters["rm_vmin"], 1e-9)
    rgba = plt.get_cmap(parameters["rm_color_map"])(np.clip((decibels - parameters["rm_vmin"]) / span, 0, 1))
    rgba[..., 3] = gain > 0
    # Row 0 of the radio map is the southern edge (lowest y); an image starts at the top.
    return _png(rgba[::-1])


def _load_sionna() -> None:
    # Mitsuba's variant is per thread, so import and compute on the same thread.
    global rt, mi, np, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_
    import mitsuba as mi_
    import numpy as np_
    import sionna.rt as rt_

    rt, mi, np, plt = rt_, mi_, np_, plt_


GPU = ThreadPoolExecutor(max_workers=1, initializer=_load_sionna)


def _health() -> dict:
    variant = mi.variant()
    return {"status": "ok", "version": VERSION, "variant": variant, "gpu": variant.startswith("cuda"), "storage": _storage()}


def _storage() -> dict:
    """What the runner keeps on this machine: the open scene's files."""
    files = [path for path in DATA_DIR.rglob("*") if path.is_file()]
    return {"bytes": sum(path.stat().st_size for path in files), "files": len(files), "scenes": len(SCENES)}


def _clear_storage() -> None:
    """Deletes all working files, also the per-start folders (``sionna-runner-*``) of earlier versions."""
    SCENES.clear()
    GEO.clear()
    for folder in [DATA_DIR, *WORK_ROOT.glob("sionna-runner-*")]:
        shutil.rmtree(folder, ignore_errors=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _bounds(scene) -> tuple:
    # Plain floats: Mitsuba points do not accept NumPy scalars.
    bbox = scene.mi_scene.bbox()
    return bbox.min.numpy().ravel().tolist(), bbox.max.numpy().ravel().tolist()


def _radio_map_area(scene, geo: dict | None) -> tuple:
    """Lowest and highest scene point under the radio map: the area of interest, else the whole scene."""
    low, high = _bounds(scene)
    aoi = geo and geo.get("aoi")
    if aoi:
        origin = geo["origin"]
        low = [aoi[0] - origin[0], aoi[1] - origin[1], low[2]]
        high = [aoi[2] - origin[0], aoi[3] - origin[1], high[2]]
    return low, high


def _corners(geo: dict, low: list, high: list) -> list:
    # MapLibre image source order: top left, top right, bottom right, bottom left.
    return [_lonlat(geo, low[0], high[1]), _lonlat(geo, high[0], high[1]), _lonlat(geo, high[0], low[1]), _lonlat(geo, low[0], low[1])]


def _render(scene, parameters: dict, radio_map=None, paths=None) -> str:
    low, high = _bounds(scene)
    center = [(low[axis] + high[axis]) / 2 for axis in range(3)]
    span = max(high[0] - low[0], high[1] - low[1], 1.0)
    if parameters["view"] == "top":
        # A tiny y offset: looking exactly down has no defined "up".
        position = [center[0], center[1] - 1e-3 * span, high[2] + 1.25 * span]
    else:
        position = [center[0], center[1] - span, low[2] + 0.9 * span]
    figure = scene.render(
        camera=rt.Camera(position=position, look_at=[center[0], center[1], low[2]]),
        radio_map=radio_map,
        paths=paths,
        rm_vmin=parameters["rm_vmin"],
        rm_vmax=parameters["rm_vmax"],
        rm_cmap=parameters["rm_color_map"],
        rm_show_color_bar=radio_map is not None,
        resolution=(960, 720),
        num_samples=32,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _built_in_scenes() -> dict[str, str]:
    return {name: path for name in dir(rt.scene) if isinstance(path := getattr(rt.scene, name), str) and path.endswith(".xml")}


def _local_scenes() -> dict[str, dict]:
    """Scene list of the panel, by id: sample areas, the sample data folder, then Sionna's built-in scenes."""
    listed = {key: {"id": key, "name": area[0], "kind": "area"} for key, area in SAMPLE_AREAS.items()}
    if LOCAL_DATA.is_dir():
        for kind, pattern in (("cityjson", "*.city.json*"), ("mitsuba", "*.xml")):
            for path in sorted(LOCAL_DATA.rglob(pattern)):
                relative = path.relative_to(LOCAL_DATA).as_posix()
                listed[relative] = {"id": relative, "name": relative, "kind": kind, "path": path}
    for name, path in _built_in_scenes().items():
        listed[f"builtin:{name}"] = {"id": f"builtin:{name}", "name": name, "kind": "builtin", "path": Path(path)}
    return listed


def _load_scene(scene_id: str, xml_path: Path, name: str) -> dict:
    scene = rt.load_scene(str(xml_path))
    SCENES.clear()  # one scene at a time: GPU memory
    GEO.clear()
    SCENES[scene_id] = scene
    low, high = _bounds(scene)
    geo = _georeference(xml_path)
    if geo is not None:
        GEO[scene_id] = geo
    return {
        "georeference": geo
        and {
            "crs": geo["crs"],
            "center": _lonlat(geo, (low[0] + high[0]) / 2, (low[1] + high[1]) / 2),
            "buildings": _buildings(geo),
            # The area of interest (radio map area) as top left, top right, bottom right, bottom left.
            "aoi": _corners(geo, *_radio_map_area(scene, geo)) if geo.get("aoi") else None,
        },
        "id": scene_id,
        "name": name,
        "objects": len(scene.objects),
        "bounds": {"min": low, "max": high},
        # A scene on the map is seen there; only scenes without a georeference need a picture.
        "image": _render(scene, DEFAULTS) if geo is None else None,
    }


def _local(scene_id: str, lon: float, lat: float) -> dict:
    scene, geo = SCENES.get(scene_id), GEO.get(scene_id)
    if scene is None or geo is None:
        raise HTTPException(404, "Scene not loaded, or it has no georeference.")
    x, y = geo["from_wgs84"].transform(lon, lat)
    x, y = x - geo["origin"][0], y - geo["origin"][1]
    # The roof or ground under the point: a ray straight down from above the scene.
    low, high = _bounds(scene)
    top = high[2] + 1.0
    distance = float(scene.mi_scene.ray_intersect(mi.Ray3f(mi.Point3f(x, y, top), mi.Vector3f(0, 0, -1))).t.numpy()[0])
    return {"x": x, "y": y, "z": top - distance if math.isfinite(distance) else low[2]}


def _add_devices(scene, kind, prefix: str, devices: list[dict]) -> None:
    for index, device in enumerate(devices):
        orientation = [math.radians(device.get("yaw_deg", 0.0)), math.radians(device.get("pitch_deg", 0.0)), 0.0]
        position = [float(value) for value in device["position"]]
        scene.add(kind(name=f"{prefix}-{index}", position=position, orientation=orientation))


def _array(parameters: dict, prefix: str):
    return rt.PlanarArray(
        num_rows=parameters[f"{prefix}_num_rows"],
        num_cols=parameters[f"{prefix}_num_cols"],
        vertical_spacing=parameters[f"{prefix}_vertical_spacing"],
        horizontal_spacing=parameters[f"{prefix}_horizontal_spacing"],
        pattern=parameters[f"{prefix}_pattern"],
        polarization=parameters[f"{prefix}_polarization"],
    )


def _path_type(steps: list[str]) -> str:
    """One class per path for the map: the rarest interaction on it wins."""
    for name in ("diffraction", "scattering", "refraction", "reflection"):
        if name in steps:
            return name
    return "los"


def _path_features(paths, geo: dict, transmitters: list[dict], receivers: list[dict]) -> list[dict]:
    """Every valid path as a 3D line (lon, lat, height above ground), strongest first."""
    vertices, kinds, valid, tau = (array.numpy() for array in (paths.vertices, paths.interactions, paths.valid, paths.tau))
    if not paths.synthetic_array:
        # Per antenna element: the map shows the first element of each device.
        vertices, kinds = vertices[:, :, 0, :, 0], kinds[:, :, 0, :, 0]
        valid, tau = valid[:, 0, :, 0], tau[:, 0, :, 0]
    real, imag = (part.numpy() for part in paths.a)
    with np.errstate(divide="ignore"):
        gain = 10 * np.log10((real**2 + imag**2).mean(axis=(1, 3)))  # [rx, tx, path]

    def point(x: float, y: float, z: float) -> list[float]:
        return [*_lonlat(geo, x, y), round(float(z), 2)]

    features = []
    for rx, tx, index in zip(*np.nonzero(valid)):
        if not math.isfinite(gain[rx, tx, index]):
            continue
        codes = kinds[:, rx, tx, index].tolist()
        depth = codes.index(0) if 0 in codes else len(codes)
        steps = [INTERACTIONS.get(code, "other") for code in codes[:depth]]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        point(*transmitters[tx]["position"]),
                        *(point(*vertex) for vertex in vertices[:depth, rx, tx, index].tolist()),
                        point(*receivers[rx]["position"]),
                    ],
                },
                "properties": {
                    "tx": f"tx-{tx}",
                    "rx": f"rx-{rx}",
                    "type": _path_type(steps),
                    "interactions": " → ".join(steps) or "line of sight",
                    "order": depth,
                    "gain_db": round(float(gain[rx, tx, index]), 1),
                    "delay_ns": round(float(tau[rx, tx, index]) * 1e9, 1),
                },
            }
        )
    features.sort(key=lambda feature: -feature["properties"]["gain_db"])
    return features[:MAX_PATHS]


def _run(scene_id: str, body: dict) -> dict:
    scene = SCENES.get(scene_id)
    if scene is None:
        raise HTTPException(404, "Scene not loaded; upload it again.")
    parameters = {**DEFAULTS, **body.get("parameters", {})}
    transmitters, receivers = body.get("transmitters", []), body.get("receivers", [])
    started = time.time()

    for name in [*scene.transmitters, *scene.receivers]:
        scene.remove(name)
    scene.frequency = parameters["frequency_ghz"] * 1e9
    scene.tx_array, scene.rx_array = _array(parameters, "tx"), _array(parameters, "rx")
    _add_devices(scene, rt.Transmitter, "tx", transmitters)
    _add_devices(scene, rt.Receiver, "rx", receivers)

    result: dict = {}
    radio_map = paths = None
    if body.get("radio_map") and transmitters:
        # As the desktop GUI: above the scene's lowest point, over the whole scene or only its area of interest.
        low, high = _radio_map_area(scene, GEO.get(scene_id))
        size = [high[0] - low[0], high[1] - low[1]]
        cell_size = [parameters["rm_cell_size_x"], parameters["rm_cell_size_y"]]
        cells = (size[0] / cell_size[0]) * (size[1] / cell_size[1])
        if cells > MAX_RADIO_MAP_CELLS:
            raise HTTPException(400, f"Radio map would have {cells:.0f} cells (limit {MAX_RADIO_MAP_CELLS}); use a larger cell size.")
        radio_map = rt.RadioMapSolver()(
            scene,
            center=[(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] + parameters["rm_height_above_ground"]],
            orientation=[0.0, 0.0, 0.0],
            size=size,
            cell_size=cell_size,
            samples_per_tx=int(10 ** parameters["rm_log_samples"]) // len(transmitters),
            max_depth=parameters["rm_max_depth"],
            **{name: parameters[f"rm_{name}"] for name in FEATURES},
        )
        gain = radio_map.path_gain.numpy().max(axis=0)
        covered = gain > 0
        decibels = 10 * np.log10(gain[covered])
        result["radio_map"] = {
            "cells": [int(gain.shape[1]), int(gain.shape[0])],
            "coverage": float(covered.mean()),
            "min_db": float(decibels.min()) if decibels.size else None,
            "max_db": float(decibels.max()) if decibels.size else None,
            "median_db": float(np.median(decibels)) if decibels.size else None,
        }
        # The colour scale from vmin (left) to vmax (right).
        result["radio_map"]["colorbar"] = _png(plt.get_cmap(parameters["rm_color_map"])(np.linspace(0, 1, 256))[None])
        geo = GEO.get(scene_id)
        if geo is not None:
            result["radio_map"]["image"] = _radio_map_png(gain, parameters)
            result["radio_map"]["corners"] = _corners(geo, low, high)
    if body.get("paths") and transmitters and receivers:
        paths = rt.PathSolver()(
            scene,
            max_depth=parameters["paths_max_depth"],
            max_num_paths_per_src=1_000_000,
            samples_per_src=1_000_000,
            synthetic_array=parameters["paths_synthetic_array"],
            seed=12345,
            **{name: parameters[f"paths_{name}"] for name in FEATURES},
        )
        result["paths"] = {"count": int(np.count_nonzero(paths.valid.numpy()))}
        if result["paths"]["count"] == 0:
            paths = None  # Sionna's renderer fails on an empty set of paths
        if paths is not None and GEO.get(scene_id) is not None:
            result["paths"]["features"] = {
                "type": "FeatureCollection",
                "features": _path_features(paths, GEO[scene_id], transmitters, receivers),
            }
    geo = GEO.get(scene_id)
    if geo is not None:
        # Map position and height above ground (the scene's z).
        result["devices"] = {
            kind: [[*_lonlat(geo, *device["position"][:2]), float(device["position"][2])] for device in devices]
            for kind, devices in (("transmitters", transmitters), ("receivers", receivers))
        }
    else:
        result["image"] = _render(scene, parameters, radio_map, paths)
    result["seconds"] = round(time.time() - started, 2)
    return result


# -- HTTP -------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _clear_storage()
    yield
    _clear_storage()


app = FastAPI(title="TOSCA-Sionna Runner", version=VERSION, lifespan=lifespan)


@app.middleware("http")
async def guard(request: Request, call_next):
    # Any web page can reach 127.0.0.1, and a multipart POST needs no CORS
    # preflight, so the Origin is checked here and not only by the browser.
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "Origin not allowed."}, status_code=403)
    if TOKEN and request.method != "OPTIONS":
        sent = request.headers.get("authorization", "")
        if not hmac.compare_digest(sent.encode(), f"Bearer {TOKEN}".encode()):
            return JSONResponse({"detail": "Invalid token."}, status_code=401)
    return await call_next(request)


# Added last so it wraps the guard and its 401/403 answers carry CORS headers.
# allow_private_network: a public TOSCA site (https://…) calling this runner on the
# user's own machine is a "private network" request; Chrome's preflight asks for it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)


async def _on_gpu(function, *arguments):
    try:
        return await asyncio.get_running_loop().run_in_executor(GPU, function, *arguments)
    except HTTPException:
        raise
    except Exception as error:  # Sionna / Mitsuba / Dr.Jit errors are the user's answer
        raise HTTPException(422, f"{type(error).__name__}: {error}") from error


@app.get("/health")
async def health() -> dict:
    return await _on_gpu(_health)


@app.get("/models")
async def models() -> list[dict]:
    return [
        {
            "key": "sionna-rt",
            "title": "Sionna RT radio propagation",
            "version": VERSION,
            "parameter_schema": PARAMETER_SCHEMA,
            "default_parameters": DEFAULTS,
        }
    ]


def _safe_target(root: Path, relative: str) -> Path | None:
    """Where an uploaded file goes; ``None`` for file types a scene does not need."""
    path = PurePosixPath(relative.replace("\\", "/"))
    # ":" covers Windows drives ("C:/x") and alternate data streams.
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in relative:
        raise HTTPException(400, f"Unsafe file path: {relative!r}")
    if path.suffix.lower() not in SCENE_SUFFIXES:
        return None
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise HTTPException(400, f"Unsafe file path: {relative!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@app.post("/scenes")
async def upload_scene(request: Request) -> dict:
    """A Mitsuba scene as one ``.zip``, or as its files named by relative path (``scene.xml``, ``meshes/a.ply``)."""
    form = await request.form(max_files=100_000, max_fields=10)
    uploads = [item for item in form.getlist("files") if hasattr(item, "filename")]
    if not uploads:
        raise HTTPException(400, "No files uploaded.")
    scene_id = uuid.uuid4().hex[:12]
    root = DATA_DIR / scene_id
    root.mkdir(parents=True)
    try:
        if len(uploads) == 1 and (uploads[0].filename or "").lower().endswith(".zip"):
            with zipfile.ZipFile(uploads[0].file) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    target = _safe_target(root, member.filename)
                    if target is not None:
                        with archive.open(member) as source, open(target, "wb") as sink:
                            shutil.copyfileobj(source, sink)
        else:
            for upload in uploads:
                target = _safe_target(root, upload.filename or "")
                if target is not None:
                    with open(target, "wb") as sink:
                        shutil.copyfileobj(upload.file, sink)
        scenes = sorted(root.rglob("*.xml"), key=lambda path: (len(path.parts), path.name))
        if scenes:
            result = await _on_gpu(_load_scene, scene_id, scenes[0], scenes[0].stem)
        else:
            city = sorted([*root.rglob("*.jsonl"), *root.rglob("*.json")])
            if not city:
                raise HTTPException(400, "No Mitsuba scene (.xml) or CityJSON (.json, .jsonl) in the upload.")
            result = await _load_city(scene_id, city[0])
    except zipfile.BadZipFile:
        shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(400, "The .zip file cannot be read.") from None
    except HTTPException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    _keep_only(scene_id)
    return result


async def _load_city(scene_id: str, city: Path | list[Path], name: str | None = None, **area) -> dict:
    """Geodata instead of a ready scene: build the Mitsuba scene from it, then load that."""
    target = DATA_DIR / scene_id / "converted"
    try:
        await asyncio.to_thread(convert, city, target, **area)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise HTTPException(400, f"{name or city.name} is not usable CityJSON: {error}") from error
    return await _on_gpu(_load_scene, scene_id, target / "scene.xml", name or city.name.split(".")[0])


def _keep_only(scene_id: str) -> None:
    for other in DATA_DIR.iterdir():
        if other.name != scene_id:
            shutil.rmtree(other, ignore_errors=True)


@app.delete("/scenes")
async def delete_scenes() -> dict:
    """Closes the open scene and deletes every working file; results were never stored."""
    await _on_gpu(_clear_storage)
    return _storage()


@app.get("/scenes")
async def scenes() -> list[dict]:
    """The scenes that are already on this machine: sample data folder and Sionna's built-in ones."""
    listed = await _on_gpu(_local_scenes)
    return [{name: entry[name] for name in ("id", "name", "kind")} for entry in listed.values()]


@app.post("/scenes/local")
async def open_local_scene(body: dict) -> dict:
    source = body.get("tile_source")
    if source is not None and not (isinstance(source, str) and source.startswith(("http://", "https://"))):
        raise HTTPException(400, "tile_source: must be an http(s) URL")
    # Looked up in the list, never joined to a path: the id cannot leave the data folder.
    entry = (await _on_gpu(_local_scenes)).get(str(body.get("id")))
    if entry is None:
        raise HTTPException(404, "No such scene in the local data folder.")
    if entry["kind"] == "area":
        name, lon, lat, side, buffer = SAMPLE_AREAS[entry["id"]]
        x, y = Transformer.from_crs("EPSG:4326", lod3_tiles.CRS, always_xy=True).transform(lon, lat)
        return await _open_area((x - side / 2, y - side / 2, x + side / 2, y + side / 2), buffer, name, source)
    scene_id = uuid.uuid4().hex[:12]
    if entry["kind"] == "cityjson":
        result = await _load_city(scene_id, entry["path"])
    else:
        result = await _on_gpu(_load_scene, scene_id, entry["path"], entry["path"].stem)
    _keep_only(scene_id)
    return result


@app.post("/scenes/area")
async def open_area(body: dict) -> dict:
    """A scene of the LoD3 buildings around an area of interest: the radio map covers the area,
    the buildings within ``buffer_m`` around it still block and reflect the signal."""
    errors = [f"{'.'.join(map(str, error.absolute_path)) or 'request'}: {error.message}" for error in Draft202012Validator(AREA_SCHEMA).iter_errors(body)]
    if errors:
        raise HTTPException(400, errors)
    west, south, east, north = body["bbox"]
    buffer = body.get("buffer_m", AREA_SCHEMA["properties"]["buffer_m"]["default"])
    to_crs = Transformer.from_crs("EPSG:4326", lod3_tiles.CRS, always_xy=True)
    xs, ys = to_crs.transform([west, east, east, west], [south, south, north, north])
    aoi = (min(xs), min(ys), max(xs), max(ys))
    width, height = aoi[2] - aoi[0], aoi[3] - aoi[1]
    if not (10 <= width <= MAX_AREA_SIDE and 10 <= height <= MAX_AREA_SIDE):
        raise HTTPException(400, f"The area is {width:.0f} × {height:.0f} m; each side must be 10 to {MAX_AREA_SIDE:.0f} m.")
    return await _open_area(aoi, buffer, f"Area {width / 1000:.2f} × {height / 1000:.2f} km", body.get("tile_source"))


async def _open_area(aoi: tuple[float, float, float, float], buffer: float, name: str, source: str | None) -> dict:
    """Opens the LoD3 buildings of ``aoi`` (x/y min, max in the tiles' CRS) plus ``buffer`` metres around it,
    from the CityJSON endpoint ``source`` (default ``SIONNA_TILE_SOURCE``)."""
    source = (source or lod3_tiles.SOURCE).rstrip("/")
    if not source:
        raise HTTPException(400, "No Sionna CityJSON endpoint: set VITE_SIONNA_CITYJSON_URL on the TOSCA site or SIONNA_TILE_SOURCE for the runner.")
    bbox = (aoi[0] - buffer, aoi[1] - buffer, aoi[2] + buffer, aoi[3] + buffer)
    try:
        tiles = await asyncio.to_thread(lod3_tiles.fetch, bbox, source)
    except OSError as error:
        raise HTTPException(502, f"LoD3 tiles cannot be downloaded from {source}: {error}") from error
    if not tiles:
        raise HTTPException(400, "No LoD3 buildings there: the area is outside the Hamburg tiles.")
    scene_id = uuid.uuid4().hex[:12]
    result = await _load_city(scene_id, tiles, name, bbox=bbox, aoi=aoi)
    _keep_only(scene_id)
    return result


@app.get("/scenes/{scene_id}/local")
async def local(scene_id: str, lon: float, lat: float) -> dict:
    """A map position in the scene's own metres, with the height of the surface there."""
    return await _on_gpu(_local, scene_id, lon, lat)


@app.post("/scenes/{scene_id}/run")
async def run(scene_id: str, body: dict) -> dict:
    errors = sorted(Draft202012Validator(RUN_SCHEMA).iter_errors(body), key=lambda error: list(error.absolute_path))
    if errors:
        # "name: message", the same shape TOSCA's backend uses for parameter errors.
        raise HTTPException(
            400, [f"{'.'.join(str(part) for part in error.absolute_path if part != 'parameters') or 'request'}: {error.message}" for error in errors]
        )
    return await _on_gpu(_run, scene_id, body)
