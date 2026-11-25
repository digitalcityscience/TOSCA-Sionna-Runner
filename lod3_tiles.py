"""Hamburg LoD3 tiles (CityJSONSeq) from a CityJSON endpoint, downloaded once into a local cache.

The endpoint (the "Sionna CityJSON endpoint") comes from the TOSCA site with each
request, or from ``SIONNA_TILE_SOURCE`` on this machine. It serves ``metadata.json``
and ``tiles/<name>.city.jsonl``.

The collection covers the Hamburg bbox in a grid of ``tile_matrix`` [columns, rows]
tiles named ``hamburg_<column>_<row>``: columns count from the west, rows from the
north. Empty tiles are not published. Only the standard library: no GDAL.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

SOURCE = os.environ.get("SIONNA_TILE_SOURCE", "").rstrip("/")
CACHE = Path(os.environ.get("SIONNA_TILE_CACHE") or Path.home() / ".cache" / "tosca-sionna-runner" / "tiles")
CRS = "EPSG:25832"  # of the collection's tiles and region_extent
# A building lies in the tile of its centre; one near a tile edge reaches this far into the next tile.
EDGE = 100.0


def _download(url: str, target: Path) -> bool:
    """False when the catalog has no such file (an empty tile)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as sink:
            while chunk := response.read(1 << 20):
                sink.write(chunk)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        if error.code == 404:
            return False
        raise
    except BaseException:
        partial.unlink(missing_ok=True)  # never leave half a tile that looks complete
        raise
    partial.replace(target)
    return True


def cache(source: str) -> Path:
    """One cache folder per endpoint: two endpoints may publish tiles with the same names."""
    return CACHE / re.sub(r"[^A-Za-z0-9.-]+", "_", source.split("://", 1)[-1]).strip("_")


def grid(source: str) -> tuple[list[float], list[int]]:
    """``region_extent`` (CRS bbox) and ``tile_matrix`` [columns, rows] of the collection."""
    metadata = cache(source) / "metadata.json"
    if not metadata.is_file() and not _download(f"{source}/metadata.json", metadata):
        raise FileNotFoundError(f"No metadata.json at {source}")
    described = json.loads(metadata.read_text(encoding="utf-8"))
    return described["region_extent"], described["tile_matrix"]


def names(bbox: tuple[float, float, float, float], source: str) -> list[str]:
    """Tiles that may hold a building inside ``bbox`` (CRS x, y)."""
    (west, south, east, north), (columns, rows) = grid(source)
    if bbox[2] < west or bbox[0] > east or bbox[3] < south or bbox[1] > north:
        return []
    width, height = (east - west) / columns, (north - south) / rows
    first_column = max(0, math.floor((bbox[0] - EDGE - west) / width))
    last_column = min(columns - 1, math.floor((bbox[2] + EDGE - west) / width))
    first_row = max(0, math.floor((north - bbox[3] - EDGE) / height))
    last_row = min(rows - 1, math.floor((north - bbox[1] + EDGE) / height))
    return [
        f"hamburg_{column}_{row}"
        for column in range(first_column, last_column + 1)
        for row in range(first_row, last_row + 1)
    ]


def fetch(bbox: tuple[float, float, float, float], source: str) -> list[Path]:
    """Local copies of the tiles around ``bbox``, downloaded the first time."""
    paths = []
    for name in names(bbox, source):
        path = cache(source) / f"{name}.city.jsonl"
        if path.is_file() or _download(f"{source}/tiles/{name}.city.jsonl", path):
            paths.append(path)
    return paths
