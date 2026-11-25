"""
Raster fetchers. Each takes the layer spec and a target grid (bounds in EPSG:3310, square pixels of `res` metres) and
returns (array[bands, h, w] in the layer's dtype with its nodata, metadata dict), or (None, meta) when the source
has nothing there. Fetchers read only the source pixels they need, at the coarsest resolution that still covers the
target (COG overviews or server-side scaling), so the same code serves zoomed-out map views and full-resolution
analysis.
"""
import math
from functools import lru_cache

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from backend.geodata.config import CRS, UA

RS = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "average": Resampling.average,
      "mode": Resampling.mode}


def _empty(spec, shape):
    return np.full((spec.get("bands_n", 1),) + shape, spec["nodata"], dtype=spec["dtype"])


def _warp_from(src, bands, dst, dst_transform, resampling, nodata, src_nodata=None):
    """Warp the part of `src` under dst's grid into dst (bands, h, w), filling only pixels still at nodata, so
    sources given newest-first win. When the target is coarser than the source, the read is decimated (GDAL then
    uses the COG's overviews), so a zoomed-out tile does not pull full-resolution pixels."""
    nb, h, w = dst.shape
    l, t = dst_transform.c, dst_transform.f
    r, b = l + dst_transform.a * w, t + dst_transform.e * h
    try:
        sl, sb, sr, st = transform_bounds(CRS, src.crs, l, b, r, t, densify_pts=11)
    except Exception:  # noqa: BLE001
        return False
    full = Window(0, 0, src.width, src.height)
    win = from_bounds(sl, sb, sr, st, src.transform)
    if not _overlaps(win, src):
        return False
    win = Window(math.floor(win.col_off) - 1, math.floor(win.row_off) - 1, math.ceil(win.width) + 3,
                 math.ceil(win.height) + 3).intersection(full)
    if win.width < 1 or win.height < 1:
        return False
    dst_px = (sr - sl) / w                      # target pixel size in source units
    dec = max(1, int(dst_px / abs(src.transform.a)))
    oh, ow = max(1, math.ceil(win.height / dec)), max(1, math.ceil(win.width / dec))
    src_nodata = src.nodata if src_nodata is None else src_nodata
    read_rs = Resampling.nearest if resampling in (Resampling.nearest, Resampling.mode) else Resampling.average
    ok = False
    for k, bi in enumerate(bands):
        a = src.read(bi, window=win, out_shape=(oh, ow), resampling=read_rs)
        st_ = src.window_transform(win) * rasterio.Affine.scale(win.width / ow, win.height / oh)
        tmp = np.full((h, w), nodata, dtype=dst.dtype)
        reproject(a, tmp, src_transform=st_, src_crs=src.crs, src_nodata=src_nodata, dst_transform=dst_transform,
                  dst_crs=CRS, dst_nodata=nodata, resampling=resampling)
        m = (dst[k] == nodata) & (tmp != nodata)
        if m.any():
            dst[k][m] = tmp[m]
            ok = True
    return ok


def _overlaps(win, src):
    return not (win.col_off >= src.width or win.row_off >= src.height or win.col_off + win.width <= 0
                or win.row_off + win.height <= 0)


def _grid(bounds, res):
    l, b, r, t = bounds
    w, h = int(round((r - l) / res)), int(round((t - b) / res))
    return from_origin(l, t, res, res), (h, w)


# ----------------------------------------------------------------------------- LANDFIRE GeoServer WCS (EPSG:5070)
LF_WCS = "https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/ows"
LF_ENVELOPE = (-2362425.0 + 90, 258945.0 + 90, 2263815.0 - 90, 3267405.0 - 90)  # EPSG:5070 CONUS extent, 3 px inside


def landfire_wcs(spec, bounds, res):
    tr, shape = _grid(bounds, res)
    x0, y0, x1, y1 = transform_bounds(CRS, "EPSG:5070", *bounds, densify_pts=21)
    pad = max(60.0, 2 * res)
    # clip to the CONUS coverage (requests past its west edge make the server fail)
    x0, y0 = max(x0 - pad, LF_ENVELOPE[0]), max(y0 - pad, LF_ENVELOPE[1])
    x1, y1 = min(x1 + pad, LF_ENVELOPE[2]), min(y1 + pad, LF_ENVELOPE[3])
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None, {}
    params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
              ("coverageId", f"landfire_wcs__{spec['src']['coverage']}_CONUS"), ("format", "image/tiff"),
              ("subset", f"X({x0},{x1})"), ("subset", f"Y({y0},{y1})")]
    if res > 45:  # let the server scale down; native is 30 m
        params.append(("scalesize", f"i({max(1, int((x1 - x0) / res))}),j({max(1, int((y1 - y0) / res))})"))
    r = requests.get(LF_WCS, params=params, headers=UA, timeout=300)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/tiff"):
        raise RuntimeError(r.text[:200])
    out = _empty(spec, shape)
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        a = src.read(1)
        a[(a == 32767) | (a == -9999)] = spec["nodata"]
        tmp = np.full(shape, spec["nodata"], dtype=spec["dtype"])
        reproject(a.astype(spec["dtype"]), tmp, src_transform=src.transform, src_crs=src.crs, src_nodata=spec["nodata"],
                  dst_transform=tr, dst_crs=CRS, dst_nodata=spec["nodata"], resampling=Resampling.nearest)
    out[0] = tmp
    return (out if (out != spec["nodata"]).any() else None), {"source_version": spec["src"]["coverage"]}


# ----------------------------------------------------------------------------- ArcGIS ImageServer exportImage
def arcgis_export(spec, bounds, res):
    tr, (h, w) = _grid(bounds, res)
    r = requests.get(spec["src"]["url"] + "/exportImage", headers=UA, timeout=180, params={
        "bbox": ",".join(map(str, bounds)), "bboxSR": 3310, "imageSR": 3310, "size": f"{w},{h}", "format": "tiff",
        "pixelType": spec["src"].get("pixel_type", "U16"), "interpolation": "RSP_NearestNeighbor", "f": "image"})
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/tiff"):
        raise RuntimeError(r.text[:200])
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        a = src.read(1, out_shape=(h, w))
    a = a.astype(spec["dtype"])[None]
    return (a if a.any() else None), {}


# ----------------------------------------------------------------------------- Planetary Computer STAC (COGs)
@lru_cache(maxsize=1)
def _pc():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def _search(collection, bbox4326, **kw):
    for attempt in range(3):
        try:
            return list(_pc().search(collections=[collection], bbox=bbox4326, **kw).items())
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            _pc.cache_clear()


def stac_cog(spec, bounds, res):
    s = spec["src"]
    tr, shape = _grid(bounds, res)
    bbox = transform_bounds(CRS, "EPSG:4326", *bounds, densify_pts=11)
    items = _search(s["collection"], bbox, **({"query": s["query"]} if s.get("query") else {}))
    if s.get("filter"):
        items = [i for i in items if s["filter"](i)]
    if not items:
        return None, {}
    # newest first, so newer acquisitions win where items overlap; older ones fill gaps
    items.sort(key=lambda i: i.properties.get("datetime") or i.properties.get("start_datetime") or "", reverse=True)
    out = _empty(spec, shape)
    used = []
    for it in items:
        href = it.assets[s["asset"]].href
        with rasterio.open(href) as src:
            if _warp_from(src, s.get("bands", [1]), out, tr, RS[spec.get("resampling", "bilinear")],
                          spec["nodata"], src_nodata=s.get("src_nodata")):
                used.append((it.properties.get("datetime") or it.properties.get("start_datetime") or "")[:10])
        if not (out == spec["nodata"]).any():
            break
    if not used:
        return None, {}
    return out, {"dates": sorted(set(used))}


# ----------------------------------------------------------------------------- Sentinel-2 spectral index
def s2_index(spec, bounds, res):
    """Clearest Sentinel-2 L2A scene(s) in the layer's date window; index = (a - b) / (a + b) x 10000."""
    s = spec["src"]
    tr, shape = _grid(bounds, res)
    bbox = transform_bounds(CRS, "EPSG:4326", *bounds, densify_pts=11)
    items = _search("sentinel-2-l2a", bbox, datetime=s["window"], query={"eo:cloud_cover": {"lt": 20}})
    if not items:
        return None, {}
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    bands = {}
    used = []
    for name in (s["a"], s["b"], "SCL"):
        arr = np.full((1,) + shape, 0, dtype="uint16")
        for it in items[:6]:
            with rasterio.open(it.assets[name].href) as src:
                if _warp_from(src, [1], arr, tr, Resampling.nearest if name == "SCL" else Resampling.bilinear, 0):
                    if name == s["a"]:
                        used.append(it.properties["datetime"][:10])
            if not (arr == 0).any():
                break
        bands[name] = arr[0].astype("float32")
    a, b, scl = bands[s["a"]], bands[s["b"]], bands["SCL"]
    bad = (a == 0) | (b == 0) | np.isin(scl, [0, 1, 3, 8, 9, 10, 11])  # nodata, saturated, shadow, cloud, snow
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = (a - b) / (a + b)
    out = np.where(bad, spec["nodata"], np.clip(idx * 10000, -10000, 10000)).astype(spec["dtype"])[None]
    return (out if (out != spec["nodata"]).any() else None), {"dates": sorted(set(used))}


# ----------------------------------------------------------------------------- Meta / WRI canopy height (S3 quadkeys)
META_CHM = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/{}.tif"


def _quadkey(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return "".join(str((1 if x >> (k - 1) & 1 else 0) + (2 if y >> (k - 1) & 1 else 0)) for k in range(z, 0, -1))


def meta_chm(spec, bounds, res):
    tr, shape = _grid(bounds, res)
    w, s, e, n = transform_bounds(CRS, "EPSG:4326", *bounds, densify_pts=11)
    keys = sorted({_quadkey(lo, la, 9) for lo in np.linspace(w, e, 4) for la in np.linspace(s, n, 4)})
    out = _empty(spec, shape)
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for k in keys:
            try:
                with rasterio.open(META_CHM.format(k)) as src:
                    _warp_from(src, [1], out, tr, Resampling.bilinear, spec["nodata"], src_nodata=255)
            except rasterio.errors.RasterioIOError:
                continue
    return (out if (out != spec["nodata"]).any() else None), {}


# ----------------------------------------------------------------------------- local GeoTIFF (derived products)
def local_file(spec, bounds, res):
    from backend.geodata.config import DERIVED
    tr, shape = _grid(bounds, res)
    p = DERIVED / spec["src"]["file"]
    if not p.exists():
        raise FileNotFoundError(f"{p} not built yet (python -m backend.geodata.build gridmet)")
    out = _empty(spec, shape)
    with rasterio.open(p) as src:
        _warp_from(src, [spec["src"]["band"]], out, tr, RS[spec.get("resampling", "nearest")], spec["nodata"])
    return (out if (out != spec["nodata"]).any() else None), {}


FETCHERS = {"landfire_wcs": landfire_wcs, "arcgis_export": arcgis_export, "stac_cog": stac_cog,
            "s2_index": s2_index, "meta_chm": meta_chm, "local_file": local_file}
