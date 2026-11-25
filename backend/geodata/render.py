"""
Web-map tiles (EPSG:3857, z/x/y, 256 px PNG) drawn from the local cache. A raster tile is read from the pyramid
level closest to the screen resolution, warped to web mercator and coloured with the layer's style. Finished tiles
are kept on disk (PNG/<layer>/<z>/<x>/<y>.png) unless a source fetch failed, so the map is fast on the second view.
"""
import io
import math
from functools import lru_cache

import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

from backend.geodata import store, vectors
from backend.geodata.catalog import layer, min_zoom
from backend.geodata.config import CRS, PNG

R = 6378137.0
STYLE_VERSION = "1"


def merc_bounds(z, x, y):
    n = 2 ** z
    s = 2 * math.pi * R / n
    left = -math.pi * R + x * s
    top = math.pi * R - y * s
    return left, top - s, left + s, top


@lru_cache(maxsize=64)
def _lut(layer_id):
    """Colour lookup for one layer: (kind, table or (cmap, vmin, vmax, below))."""
    spec = layer(layer_id)
    st = spec["style"]
    if spec["kind"] == "categorical":
        mx = max(int(k) for k in st) + 1
        lut = np.zeros((max(mx, 256), 4), dtype=np.uint8)
        for code, (_, col) in st.items():
            c = col.lstrip("#")
            lut[int(code)] = [int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 210]
        return lut
    if spec["kind"] == "continuous":
        import matplotlib
        cmap = matplotlib.colormaps[st["cmap"]]
        return (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    return None


def colorize(spec, a):
    """a: (bands, h, w) raw values -> RGBA uint8 (h, w, 4)."""
    nod = spec["nodata"]
    if spec["kind"] == "rgb":
        b = spec["style"]["bands"]
        rgb = np.stack([a[k] for k in b], -1).astype(np.uint8)
        alpha = np.where((a[b[0]] == nod) & (a[b[1]] == nod) & (a[b[2]] == nod), 0, 255).astype(np.uint8)
        return np.dstack([rgb, alpha])
    v = a[0]
    valid = v != nod
    lut = _lut(spec["id"])
    if spec["kind"] == "categorical":
        idx = np.clip(np.where(valid, v, 0).astype(np.int64), 0, len(lut) - 1)
        out = lut[idx]
        out[~valid] = 0
        return out
    st = spec["style"]
    val = v.astype("float32") * (spec["scale"] or 1.0)
    t = np.clip((val - st["vmin"]) / max(1e-12, st["vmax"] - st["vmin"]), 0, 1)
    out = lut[(t * 255).astype(np.int64)]
    out[..., 3] = np.where(valid & (val >= st.get("below", -np.inf)), 205, 0)
    return out


def raster_tile(layer_id, z, x, y, size=256):
    """(PNG bytes or None, complete: bool). complete=False when a needed source tile could not be fetched."""
    from PIL import Image
    spec = layer(layer_id)
    if z < min_zoom(spec):
        return None, True
    mb = merc_bounds(z, x, y)
    b = transform_bounds("EPSG:3857", CRS, *mb, densify_pts=21)
    web_px = (mb[2] - mb[0]) / size
    # metres on the ground per web pixel at this latitude
    lat = math.degrees(math.atan(math.sinh((mb[1] + mb[3]) / 2 / R)))
    ground_px = web_px * math.cos(math.radians(lat))
    pad = ground_px * 2
    try:
        a, tr, lev = store.read(layer_id, (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), res=ground_px)
        complete = True
    except Exception:  # noqa: BLE001
        return None, False
    if not (a != spec["nodata"]).any():
        return None, True
    out = np.full((a.shape[0], size, size), spec["nodata"], dtype=a.dtype)
    rs = store.resampling_for(spec)
    for k in range(a.shape[0]):
        reproject(a[k], out[k], src_transform=tr, src_crs=CRS, src_nodata=spec["nodata"],
                  dst_transform=from_bounds(*mb, size, size), dst_crs="EPSG:3857", dst_nodata=spec["nodata"],
                  resampling=rs)
    rgba = colorize(spec, out)
    if not rgba[..., 3].any():
        return None, complete
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=False, compress_level=4)
    return buf.getvalue(), complete


def tile_png(layer_id, z, x, y):
    """PNG bytes for z/x/y (None = transparent), using and filling the PNG cache."""
    p = PNG / layer_id / STYLE_VERSION / str(z) / str(x) / f"{y}.png"
    e = p.with_suffix(".none")
    if p.exists():
        return p.read_bytes()
    if e.exists():
        return None
    spec = layer(layer_id)
    if spec["type"] == "vector":
        img = vectors.draw(layer_id, z, x, y)
        data = None
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, "PNG", compress_level=4)
            data = buf.getvalue()
        complete = True
    else:
        data, complete = raster_tile(layer_id, z, x, y)
    if complete:
        p.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            e.write_bytes(b"")
        else:
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(p)
    return data


def clear_png(layer_id):
    import shutil
    shutil.rmtree(PNG / layer_id, ignore_errors=True)
