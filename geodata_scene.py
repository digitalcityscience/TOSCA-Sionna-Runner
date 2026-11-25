"""CityJSON / CityJSONSeq -> Mitsuba scene for Sionna RT, with its georeference.

    python geodata_scene.py tile.city.jsonl out_dir [--bbox MINX MINY MAXX MAXY]

Writes ``scene.xml``, ``meshes/<material>.ply`` and ``scene.json``. The scene is
in metres relative to ``origin`` (in the source CRS), so a local point maps back
to the map as ``origin + local``; z is height above ground. One mesh per
material: Sionna assigns radio materials per object, and a city needs no finer split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Iterator

import mapbox_earcut
import numpy as np

# Semantic surface type -> ITU material. Anything not listed is a wall.
WALL = "itu_brick"
MATERIALS = {"RoofSurface": "itu_concrete", "Window": "itu_glass", "Door": "itu_wood"}
# Sealed city ground; ITU's ground types are only defined for 1 to 10 GHz, concrete for 1 to 100 GHz.
GROUND = "itu_concrete"
GROUND_MARGIN = 50.0  # metres of ground around the buildings
# A CityJSONSeq feature's first vertex (integers before the transform), and how far it may lie from
# the building's centre: no building is larger, so a feature farther outside a bbox is not parsed.
FIRST_VERTEX = re.compile(r'"vertices":\[\[(-?\d+),(-?\d+)')
FAR = 500.0
# How many list levels sit above a surface (a list of rings) per geometry type.
SURFACE_DEPTH = {"MultiSurface": 1, "CompositeSurface": 1, "Solid": 2, "MultiSolid": 3, "CompositeSolid": 3}

# Rings of (n, 3) coordinates (x, y in the CRS, z above the building's ground),
# semantic surface type, height of the building the surface belongs to.
Surface = tuple[list[np.ndarray], str | None, float]


def _flatten(value, levels: int) -> list:
    for _ in range(levels):
        value = [item for group in value for item in (group if group is not None else [None])]
    return value


def _indices(value) -> Iterator[int]:
    if isinstance(value, list):
        for item in value:
            yield from _indices(item)
    elif value is not None:
        yield value


def _z_ranges(city_objects: dict, vertices: np.ndarray) -> dict[str, tuple[float, float]]:
    """Lowest and highest z per building; its parts and installations share them."""
    ranges: dict[str, tuple[float, float]] = {}
    roots = {}
    for key, city_object in city_objects.items():
        root = key
        while city_objects.get(root, {}).get("parents"):
            root = city_objects[root]["parents"][0]
        roots[key] = root
        used = sorted(set(_indices([g.get("boundaries") for g in city_object.get("geometry", [])])))
        if used:
            z = vertices[used, 2]
            low, high = ranges.get(root, (np.inf, -np.inf))
            ranges[root] = (min(low, z.min()), max(high, z.max()))
    return {key: ranges[root] for key, root in roots.items() if root in ranges}


def _surfaces(city_objects: dict, vertices: np.ndarray) -> Iterator[Surface]:
    z_ranges = _z_ranges(city_objects, vertices)
    for key, city_object in city_objects.items():
        geometries = [g for g in city_object.get("geometry", []) if g.get("type") in SURFACE_DEPTH]
        if not geometries:
            continue
        # No terrain model: every building stands on the flat ground at z = 0,
        # as it does on the map. Otherwise buildings on higher ground would float.
        low, high = z_ranges[key]
        best = max(str(g.get("lod", "")) for g in geometries)
        for geometry in geometries:
            if str(geometry.get("lod", "")) != best:
                continue
            depth = SURFACE_DEPTH[geometry["type"]]
            surfaces = _flatten(geometry["boundaries"], depth - 1)
            semantics = geometry.get("semantics") or {}
            types = [surface.get("type") for surface in semantics.get("surfaces", [])]
            values = _flatten(semantics.get("values") or [], depth - 1)
            for index, rings in enumerate(surfaces):
                value = values[index] if index < len(values) else None
                kind = types[value] if value is not None else None
                yield [vertices[ring] - [0.0, 0.0, low] for ring in rings], kind, float(high - low)


def read_surfaces(path: Path, bbox: tuple[float, float, float, float] | None = None) -> tuple[Iterator[Surface], str]:
    """Surfaces of a CityJSON file or a CityJSONSeq (one feature per line), and the CRS."""
    with open(path, encoding="utf-8") as file:
        header = json.loads(file.readline() if path.suffix == ".jsonl" else file.read())
    transform = header.get("transform", {"scale": [1, 1, 1], "translate": [0, 0, 0]})
    scale, translate = np.array(transform["scale"]), np.array(transform["translate"])
    crs = header.get("metadata", {}).get("referenceSystem", "")

    def far(line: str) -> bool:
        # Parsing a LoD3 tile is most of the time; a building's first vertex is near its centre.
        match = FIRST_VERTEX.search(line) if bbox is not None else None
        if match is None:
            return False
        x, y = int(match[1]) * scale[0] + translate[0], int(match[2]) * scale[1] + translate[1]
        return not (bbox[0] - FAR <= x <= bbox[2] + FAR and bbox[1] - FAR <= y <= bbox[3] + FAR)

    def features() -> Iterator[dict]:
        yield header
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as file:
                next(file)
                yield from (json.loads(line) for line in file if line.strip() and not far(line))

    def surfaces() -> Iterator[Surface]:
        for feature in features():
            if not feature.get("vertices"):
                continue
            vertices = np.array(feature["vertices"], dtype=np.float64) * scale + translate
            if bbox is not None:
                # A feature is one building; a single-file CityJSON is kept or dropped whole.
                x, y = vertices[:, 0].mean(), vertices[:, 1].mean()
                if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                    continue
            yield from _surfaces(feature["CityObjects"], vertices)

    return surfaces(), crs


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # np.cross costs more than the maths for the small arrays of one surface (LoD3: 100k+ surfaces per area).
    return np.stack(
        [a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2], a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]],
        axis=-1,
    )


def _triangulate(rings: list[np.ndarray]) -> np.ndarray:
    """Triangles (n, 3, 3) of a planar polygon with holes."""
    outer = rings[0]
    if len(outer) < 3:
        return np.empty((0, 3, 3))
    # Newell normal: robust for non-convex rings.
    normal = _cross(outer, np.roll(outer, -1, axis=0)).sum(axis=0)
    if not normal.any():
        return np.empty((0, 3, 3))
    if len(rings) == 1 and len(outer) == 3:
        return outer[None].copy()  # already a triangle; its order gives the normal
    points = np.concatenate(rings)
    keep = [axis for axis in range(3) if axis != int(np.abs(normal).argmax())]
    ends = np.cumsum([len(ring) for ring in rings]).astype(np.uint32)
    # Earcut wants small numbers: UTM northings lose precision in its float maths.
    flat = np.ascontiguousarray(points[:, keep] - points[0, keep])
    triangles = points[mapbox_earcut.triangulate_float64(flat, ends).reshape(-1, 3)]
    if len(triangles):
        facing = _cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) @ normal
        triangles[facing < 0] = triangles[facing < 0][:, ::-1]
    return triangles


def _write_ply(path: Path, triangles: np.ndarray) -> None:
    vertices = triangles.reshape(-1, 3).astype("<f4")
    faces = np.empty(len(triangles), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    faces["count"] = 3
    faces["indices"] = np.arange(len(vertices)).reshape(-1, 3)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as file:
        file.write(header.encode("ascii"))
        file.write(vertices.tobytes())
        file.write(faces.tobytes())


def convert(
    source: Path | list[Path],
    target: Path,
    bbox: tuple[float, float, float, float] | None = None,
    aoi: tuple[float, float, float, float] | None = None,
) -> dict:
    """With ``bbox``, the ground covers all of it, also where it holds no building.
    ``aoi`` (inside ``bbox``) is where the radio map goes; the rest of ``bbox`` only blocks and reflects."""
    sources = source if isinstance(source, list) else [source]
    surfaces, crs = iter(()), ""
    for path in sources:
        more, crs = read_surfaces(path, bbox)
        surfaces = itertools.chain(surfaces, more)
    by_material: dict[str, list[np.ndarray]] = {}
    # shortcut: footprints come from GroundSurface semantics, so a file without them
    # shows no buildings on the map (the simulation is unaffected); derive outlines when needed.
    buildings = []
    for rings, kind, height in surfaces:
        if kind == "GroundSurface":  # lies in the ground plane: the footprint for the map
            buildings.append({"ring": np.round(rings[0][:, :2], 2).tolist(), "height": round(height, 2)})
            continue
        triangles = _triangulate(rings)
        if len(triangles):
            by_material.setdefault(MATERIALS.get(kind, WALL), []).append(triangles)
    if not by_material and bbox is None:
        raise ValueError("No building surfaces found (wrong file, or the bbox is outside it).")
    meshes = {material: np.concatenate(parts) for material, parts in by_material.items()}

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        top = max((float(mesh[..., 2].max()) for mesh in meshes.values()), default=0.0)
    else:
        everything = np.concatenate([mesh.reshape(-1, 3) for mesh in meshes.values()])
        low, high = everything.min(axis=0), everything.max(axis=0)
        x0, y0, x1, y1 = low[0] - GROUND_MARGIN, low[1] - GROUND_MARGIN, high[0] + GROUND_MARGIN, high[1] + GROUND_MARGIN
        top = float(high[2])
    # Whole metres keep the origin exact when it is written and read back.
    origin = np.array([round((x0 + x1) / 2), round((y0 + y1) / 2), 0.0])
    # shortcut: flat ground with every building set on it, add terrain (DGM) when height differences matter.
    z = 0.0
    ground = np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]], [[x0, y0, z], [x1, y1, z], [x0, y1, z]]])
    # Joined, not replaced: the roofs share the ground's material.
    meshes[GROUND] = np.concatenate([meshes[GROUND], ground]) if GROUND in meshes else ground

    (target / "meshes").mkdir(parents=True, exist_ok=True)
    xml = ['<scene version="2.1.0">', '\t<integrator type="path"><integer name="max_depth" value="12"/></integrator>']
    xml += [
        f'\t<bsdf type="diffuse" id="mat-{material}" name="mat-{material}">'
        '<rgb value="0.8 0.8 0.8" name="reflectance"/></bsdf>'
        for material in meshes
    ]
    xml.append('\t<emitter type="constant" id="World" name="World"><rgb value="0.85 0.85 0.85" name="radiance"/></emitter>')
    for material, triangles in meshes.items():
        _write_ply(target / "meshes" / f"{material}.ply", triangles - origin)
        xml.append(
            # "geo-": Sionna keeps objects and materials in one namespace.
            f'\t<shape type="ply" id="mesh-geo-{material}" name="mesh-geo-{material}">'
            f'<string name="filename" value="meshes/{material}.ply"/>'
            f'<boolean name="face_normals" value="true"/><ref id="mat-{material}" name="bsdf"/></shape>'
        )
    xml.append("</scene>")
    (target / "scene.xml").write_text("\n".join(xml) + "\n", encoding="utf-8")

    georeference = {
        "crs": crs,
        "origin": origin.tolist(),  # CRS x, y of the scene's (0, 0); z is height above ground
        "bounds": {"min": [x0, y0, 0.0], "max": [x1, y1, top]},  # CRS coordinates
        "triangles": {material: len(triangles) for material, triangles in meshes.items()},
        "aoi": list(aoi) if aoi is not None else None,  # CRS bbox of the radio map, None: the whole scene
    }
    # Footprints (CRS x, y) and heights: the buildings as the map shows them.
    (target / "scene.json").write_text(json.dumps({**georeference, "buildings": buildings}), encoding="utf-8")
    return georeference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help=".city.json or .city.jsonl")
    parser.add_argument("target", type=Path, help="output folder")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"), help="in the source CRS")
    arguments = parser.parse_args()
    print(json.dumps(convert(arguments.source, arguments.target, tuple(arguments.bbox) if arguments.bbox else None), indent=2))
