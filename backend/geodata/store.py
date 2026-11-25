"""
Raster tile pyramid on disk. A tile is read from disk if present; otherwise it is fetched from the layer's source
(or, for prefetched layers whose finer tiles are all on disk, averaged from its four children), written atomically,
and returned. Tiles outside California are never fetched.

    get_tile(layer_id, level, i, j) -> np.ndarray (bands, 512, 512) or None (no data there)
    read(layer_id, bounds_3310, res) -> (array, transform): mosaic at the pyramid level closest to `res`
"""
import json
import logging
import math
import os
import threading
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box

from backend.geodata.catalog import layer
from backend.geodata.config import CRS, TILE, TILES, ca_shape, tile_bounds, tiles_for_bounds
from backend.geodata.sources import FETCHERS, RS

log = logging.getLogger("geodata")
_locks: dict = {}
_locks_guard = threading.Lock()
GT = dict(driver="GTiff", tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=1)


def _lock(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def tile_path(layer_id, level, i, j):
    return TILES / layer_id / f"L{level}" / f"{i}_{j}"


def in_california(spec, level, i, j):
    _, prep = ca_shape()
    return prep.intersects(box(*tile_bounds(spec["res"], level, i, j)))


def cached(layer_id, level, i, j):
    p = tile_path(layer_id, level, i, j)
    return p.with_suffix(".tif").exists() or p.with_suffix(".empty").exists()


def _load(p):
    with rasterio.open(p) as r:
        return r.read()


def _write(spec, level, i, j, arr, meta):
    p = tile_path(spec["id"], level, i, j)
    p.parent.mkdir(parents=True, exist_ok=True)
    if arr is None or not (arr != spec["nodata"]).any():
        p.with_suffix(".empty").write_text(json.dumps(meta or {}))
        return None
    l, b, r, t = tile_bounds(spec["res"], level, i, j)
    res = spec["res"] * 2 ** level
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
    with rasterio.open(tmp, "w", **GT, width=TILE, height=TILE, count=arr.shape[0], dtype=spec["dtype"], crs=CRS,
                       transform=from_origin(l, t, res, res), nodata=spec["nodata"]) as dst:
        dst.write(arr)
        if meta:
            dst.update_tags(META=json.dumps(meta)[:4000])
    os.replace(tmp, p.with_suffix(".tif"))
    return arr


def _from_children(spec, level, i, j):
    """Downsample the four level-1 children (all must be on disk)."""
    nb = spec.get("bands_n", 1)
    big = np.full((nb, 2 * TILE, 2 * TILE), spec["nodata"], dtype=spec["dtype"])
    for di in (0, 1):
        for dj in (0, 1):
            c = tile_path(spec["id"], level - 1, 2 * i + di, 2 * j + dj)
            if c.with_suffix(".tif").exists():
                big[:, dj * TILE:(dj + 1) * TILE, di * TILE:(di + 1) * TILE] = _load(c.with_suffix(".tif"))
    if not (big != spec["nodata"]).any():
        return None
    l, b, r, t = tile_bounds(spec["res"], level, i, j)
    res = spec["res"] * 2 ** level
    out = np.full((nb, TILE, TILE), spec["nodata"], dtype=spec["dtype"])
    rs = Resampling.mode if spec["kind"] == "categorical" or spec.get("resampling") == "nearest" else Resampling.average
    reproject(big, out, src_transform=from_origin(l, t, res / 2, res / 2), src_crs=CRS, src_nodata=spec["nodata"],
              dst_transform=from_origin(l, t, res, res), dst_crs=CRS, dst_nodata=spec["nodata"], resampling=rs)
    return out


def children_ready(spec, level, i, j):
    """All four children are on disk (or lie outside California, where nothing is ever fetched)."""
    return level > 0 and all(cached(spec["id"], level - 1, 2 * i + di, 2 * j + dj)
                             or not in_california(spec, level - 1, 2 * i + di, 2 * j + dj)
                             for di in (0, 1) for dj in (0, 1))


def get_tile(layer_id, level, i, j, fetch=True, rebuild=False):
    spec = layer(layer_id)
    p = tile_path(layer_id, level, i, j)
    if not rebuild:
        if p.with_suffix(".tif").exists():
            return _load(p.with_suffix(".tif"))
        if p.with_suffix(".empty").exists():
            return None
    if not fetch or not in_california(spec, level, i, j):
        return None
    with _lock((layer_id, level, i, j)):
        if not rebuild and p.with_suffix(".tif").exists():
            return _load(p.with_suffix(".tif"))
        if spec.get("prefetch") and children_ready(spec, level, i, j):
            return _write(spec, level, i, j, _from_children(spec, level, i, j), {"from": "children"})
        bounds = tile_bounds(spec["res"], level, i, j)
        t0 = time.time()
        for attempt in range(3):
            try:
                arr, meta = FETCHERS[spec["src"]["fetch"]](spec, bounds, spec["res"] * 2 ** level)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    log.warning("fetch %s L%d %d_%d failed: %s", layer_id, level, i, j, e)
                    raise
                time.sleep(2 * (attempt + 1))
        meta = dict(meta or {}, fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    seconds=round(time.time() - t0, 1))
        return _write(spec, level, i, j, arr, meta)


def level_for(spec, res):
    """Finest pyramid level whose pixels are no larger than `res` (clamped to the layer's levels)."""
    if res <= spec["res"]:
        return 0
    return min(spec["max_level"], int(math.floor(math.log2(res / spec["res"]) + 1e-9)))


def read(layer_id, bounds, res=None, level=None, fetch=True):
    """Mosaic of the tiles covering bounds (EPSG:3310) at the chosen level, cropped to bounds.
    Returns (array[bands, h, w], transform, level) with the layer's nodata where nothing is known."""
    spec = layer(layer_id)
    if level is None:
        level = level_for(spec, res or spec["res"])
    lres = spec["res"] * 2 ** level
    l, b, r, t = bounds
    # snap to the level's pixel grid
    from backend.geodata.config import X0, Y0
    l = X0 + math.floor((l - X0) / lres) * lres
    t = Y0 - math.floor((Y0 - t) / lres) * lres
    w = max(1, int(math.ceil((r - l) / lres)))
    h = max(1, int(math.ceil((t - b) / lres)))
    nb = spec.get("bands_n", 1)
    out = np.full((nb, h, w), spec["nodata"], dtype=spec["dtype"])
    for i, j in tiles_for_bounds(spec["res"], level, (l, t - h * lres, l + w * lres, t)):
        a = get_tile(layer_id, level, i, j, fetch=fetch)
        if a is None:
            continue
        tl, tb, tr, tt = tile_bounds(spec["res"], level, i, j)
        c0 = int(round((tl - l) / lres))
        r0 = int(round((t - tt) / lres))
        sc0, sr0 = max(0, -c0), max(0, -r0)
        dc0, dr0 = max(0, c0), max(0, r0)
        cw = min(TILE - sc0, w - dc0)
        rh = min(TILE - sr0, h - dr0)
        if cw > 0 and rh > 0:
            out[:, dr0:dr0 + rh, dc0:dc0 + cw] = a[:, sr0:sr0 + rh, sc0:sc0 + cw]
    return out, from_origin(l, t, lres, lres), level


def resampling_for(spec):
    return Resampling.nearest if spec["kind"] == "categorical" or spec.get("resampling") in ("nearest", "mode") \
        else RS.get(spec.get("resampling", "bilinear"), Resampling.bilinear)
