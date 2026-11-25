"""Paths and the tile grid shared by every raster layer."""
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(os.environ.get("GEODATA_ROOT", "/data/wildfire_benchmark_data/ca"))
TILES = ROOT / "tiles"        # raster pyramid: tiles/<layer>/L<level>/<i>_<j>.tif (or .empty)
VECTORS = ROOT / "vectors"    # statewide GeoParquet, EPSG:4326
DERIVED = ROOT / "derived"    # rasters we compute (gridMET fire-weather scenarios)
RAW = ROOT / "raw"            # downloads as received
PNG = ROOT / "png"            # rendered map tiles, safe to delete
BOUNDARY = ROOT / "ca_boundary_3310.gpkg"

CRS = "EPSG:3310"             # California (Teale) Albers, metres
X0, Y0 = -420_000.0, 500_000.0  # top-left corner of the tile grid; covers the state with margin
TILE = 512                    # pixels per tile side at every level
UA = {"User-Agent": "wildfire-ai-geodata/1.0"}


def tile_size_m(base_res: float, level: int) -> float:
    return TILE * base_res * (2 ** level)


def tile_bounds(base_res: float, level: int, i: int, j: int):
    s = tile_size_m(base_res, level)
    left, top = X0 + i * s, Y0 - j * s
    return (left, top - s, left + s, top)


def tiles_for_bounds(base_res: float, level: int, bounds):
    """(i, j) of every tile touching bounds (left, bottom, right, top) in EPSG:3310."""
    import math
    s = tile_size_m(base_res, level)
    l, b, r, t = bounds
    i0, i1 = math.floor((l - X0) / s), math.floor((r - X0 - 1e-6) / s)
    j0, j1 = math.floor((Y0 - t) / s), math.floor((Y0 - b - 1e-6) / s)
    return [(i, j) for j in range(j0, j1 + 1) for i in range(i0, i1 + 1)]


@lru_cache(maxsize=1)
def ca_shape():
    """California boundary (EPSG:3310), slightly buffered so coastal and border tiles are kept."""
    import geopandas as gpd
    from shapely.prepared import prep
    g = gpd.read_file(BOUNDARY).geometry.union_all().buffer(2000)
    return g, prep(g)
