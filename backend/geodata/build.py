"""
Build and warm the California cache.

    python -m backend.geodata.build vectors [fhsz fire_history ignitions treatments buildings]
    python -m backend.geodata.build gridmet                 # fire-weather layers from raw/gridmet/*.nc
    python -m backend.geodata.build prefetch fuel_model burn_probability ... [--workers 8]
    python -m backend.geodata.build prefetch --all          # every layer marked prefetch=True
    python -m backend.geodata.build pyramid fuel_model ...  # rebuild coarse levels from level 0
    python -m backend.geodata.build status

Prefetch downloads level 0 for every tile touching California, then builds the coarser levels from it, so the
zoomed-out map and the full-resolution queries agree. Lazy layers (NAIP, canopy height, Sentinel-2) are not
prefetched; their tiles are fetched when first viewed or queried.
"""
import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from backend.geodata import render, store
from backend.geodata.catalog import FW_BANDS, RASTERS, VECTOR_LAYERS, layer
from backend.geodata.config import DERIVED, RAW, TILES, ca_shape, tile_bounds, tiles_for_bounds

log = logging.getLogger("geodata")


def ca_tiles(spec, level):
    g, prep = ca_shape()
    from shapely.geometry import box
    return [(i, j) for i, j in tiles_for_bounds(spec["res"], level, g.bounds)
            if prep.intersects(box(*tile_bounds(spec["res"], level, i, j)))]


def prefetch(layer_id, workers=8, level=0):
    spec = layer(layer_id)
    todo = [t for t in ca_tiles(spec, level) if not store.cached(layer_id, level, *t)]
    total = len(ca_tiles(spec, level))
    log.info("%s L%d: %d tiles in California, %d to fetch", layer_id, level, total, len(todo))
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(store.get_tile, layer_id, level, i, j): (i, j) for i, j in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                failed += 1
            done += 1
            if done % 200 == 0 or done == len(todo):
                rate = done / max(1e-6, time.time() - t0)
                log.info("%s: %d/%d (%.1f tiles/s, %d failed, ~%.0f min left)", layer_id, done, len(todo), rate,
                         failed, (len(todo) - done) / max(rate, 1e-6) / 60)
    return failed


def pyramid(layer_id, workers=8):
    spec = layer(layer_id)
    for level in range(1, spec["max_level"] + 1):
        tiles = ca_tiles(spec, level)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda t: store.get_tile(layer_id, level, *t, rebuild=True), tiles))
        log.info("%s: built L%d (%d tiles)", layer_id, level, len(tiles))
    render.clear_png(layer_id)


# ----------------------------------------------------------------------------- gridMET fire weather
GM_VARS = ["erc", "vs", "th", "rmin", "fm100"]


def _read_nc(path):
    """Raw values as int32 with the variable's fill set to -1, plus its scale and offset."""
    import scipy.io
    f = scipy.io.netcdf_file(path, "r", mmap=False)
    name = [k for k in f.variables if k not in ("lat", "lon", "day", "crs")][0]
    v = f.variables[name]
    raw = v.data.astype(np.int32)
    if getattr(v, "_Unsigned", b"").lower() in (b"true", "true"):
        raw[raw < 0] += 65536
    raw[raw == int(getattr(v, "_FillValue", 32767))] = -1
    return (raw, float(getattr(v, "scale_factor", 1.0)), float(getattr(v, "add_offset", 0.0)),
            f.variables["lat"].data.copy(), f.variables["lon"].data.copy(), f.variables["day"].data.copy())


def build_gridmet(y0=2000, y1=2025, block=24):
    """Fire-weather scenario layers from daily gridMET. Processed in blocks of latitude rows to bound memory."""
    import rasterio
    from rasterio.transform import from_origin
    src = RAW / "gridmet"
    dc = DERIVED / "gridmet_daily"
    dc.mkdir(parents=True, exist_ok=True)
    cube, scale, lat, lon, days = {}, {}, None, None, None
    for v in GM_VARS:
        parts, dd = [], []
        for y in range(y0, y1 + 1):
            raw, sc, off, lat, lon, d = _read_nc(src / f"{v}_{y}.nc")
            parts.append(raw.astype("int32"))
            dd.append(d)
            scale[v] = (sc, off)
        a = np.concatenate(parts, 0)
        days = np.concatenate(dd)
        # daily cube kept as int16 (value x 10) for point time series
        sc, off = scale[v]
        val10 = np.where(a < 0, -32768, np.round((a * sc + off) * 10)).astype("int16")
        np.save(dc / f"{v}.npy", val10)
        cube[v] = val10
        del a, parts
        log.info("gridMET %s: %s", v, val10.shape)
    np.save(dc / "days_since_1900.npy", days)
    (dc / "grid.json").write_text(json.dumps({"lat": lat.tolist(), "lon": lon.tolist(), "scale": 0.1,
                                              "nodata": -32768, "vars": GM_VARS, "years": [y0, y1]}))
    T, H, W = cube["erc"].shape
    names = [b[0] for b in FW_BANDS]
    outb = {k: np.full((H, W), -9999.0, "float32") for k in names}
    rose = np.full((8, H, W), -9999.0, "float32")

    def f(v, r0, r1):
        a = cube[v][:, r0:r1].astype("float32") / 10
        a[cube[v][:, r0:r1] == -32768] = np.nan
        return a

    import warnings
    for r0 in range(0, H, block):
        r1 = min(H, r0 + block)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            erc, vs, th = f("erc", r0, r1), f("vs", r0, r1), f("th", r0, r1)
            p90 = np.nanpercentile(erc, 90, axis=0)
            p97 = np.nanpercentile(erc, 97, axis=0)
            fwd = erc >= p90[None]
            nfwd = fwd.sum(0).astype("float32")
            ws = np.where(fwd, vs, np.nan)
            w50, w90 = np.nanpercentile(ws, 50, axis=0), np.nanpercentile(ws, 90, axis=0)
            wmax = np.nanmax(ws, axis=0)
            sec = np.floor(((th + 22.5) % 360) / 45)
            shares = np.stack([((sec == k) & fwd).sum(0) / np.maximum(nfwd, 1) for k in range(8)]).astype("float32")
            strong = fwd & (vs >= w90[None])
            sshare = np.stack([((sec == k) & strong).sum(0) for k in range(8)])
            rmin10 = np.nanpercentile(np.where(fwd, f("rmin", r0, r1), np.nan), 10, axis=0)
            fm10 = np.nanpercentile(np.where(fwd, f("fm100", r0, r1), np.nan), 10, axis=0)
        land = ~np.isnan(p90) & (nfwd > 0)
        vals = [p90, p97, w50, w90, wmax, np.argmax(shares, 0) * 45.0, shares.max(0), np.argmax(sshare, 0) * 45.0,
                rmin10, fm10]
        for k, a in zip(names, vals):
            outb[k][r0:r1] = np.where(land, a, -9999.0)
        rose[:, r0:r1] = np.where(land[None], shares, -9999.0)
        log.info("gridMET stats rows %d-%d of %d", r0, r1, H)
    res = abs(float(lat[1] - lat[0]))
    flip = (lambda a: a) if lat[0] > lat[-1] else (lambda a: a[..., ::-1, :])
    tr = from_origin(float(lon.min()) - res / 2, float(lat.max()) + res / 2, res, res)
    prof = dict(driver="GTiff", width=W, height=H, crs="EPSG:4326", transform=tr, compress="deflate",
                dtype="float32", nodata=-9999.0)
    with rasterio.open(DERIVED / "fireweather_gridmet.tif", "w", count=len(names), **prof) as dst:
        for k, name in enumerate(names, 1):
            dst.write(flip(outb[name]), k)
            dst.set_band_description(k, name)
    with rasterio.open(DERIVED / "fireweather_windrose.tif", "w", count=8, **prof) as dst:
        dst.write(flip(rose))
        for k, sname in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], 1):
            dst.set_band_description(k, sname)
    import shutil
    for lid in names:
        render.clear_png(lid)
        shutil.rmtree(TILES / lid, ignore_errors=True)
    log.info("fire-weather layers written")


def status():
    rows = []
    for lid in RASTERS:
        spec = layer(lid)
        n0 = len(list((TILES / lid / "L0").glob("*.tif"))) if (TILES / lid / "L0").exists() else 0
        e0 = len(list((TILES / lid / "L0").glob("*.empty"))) if (TILES / lid / "L0").exists() else 0
        tot = len(ca_tiles(spec, 0)) if spec.get("prefetch") else None
        rows.append((lid, n0, e0, tot))
        print(f"{lid:24s} L0 tiles {n0:6d} (+{e0} empty)" + (f" of {tot}" if tot else "  [on demand]"))
    for lid, spec in VECTOR_LAYERS.items():
        p = layer(lid)["path"]
        from pathlib import Path
        m = Path(p).with_suffix(".meta.json")
        print(f"{lid:24s} {'built ' + m.read_text() if m.exists() else 'not built'}")
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vectors", "gridmet", "prefetch", "pyramid", "status"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "vectors":
        from backend.geodata import vectors
        for k in a.names or list(vectors.INGEST):
            t = time.time()
            n = vectors.INGEST[k]()
            log.info("%s: %s features (%.0f s)", k, n, time.time() - t)
            for lid in VECTOR_LAYERS:
                if lid.startswith(k) or (k == "buildings" and lid == "buildings"):
                    render.clear_png(lid)
    elif a.cmd == "gridmet":
        build_gridmet()
    elif a.cmd in ("prefetch", "pyramid"):
        names = [k for k, s in RASTERS.items() if s.get("prefetch")] if a.all else a.names
        for lid in names:
            if a.cmd == "prefetch":
                prefetch(lid, a.workers)
            pyramid(lid, a.workers)
    else:
        status()


if __name__ == "__main__":
    main()
