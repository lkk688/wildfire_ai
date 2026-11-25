"""
Reading the cached layers the way the agent needs them. Every function returns plain JSON-able dicts that carry the
layer's source, units and date, so an answer built on them can cite where each number came from.

    point(lon, lat)                         every layer's value at a point (+ vectors at / near it)
    area(lon, lat, radius_m=500)            summaries inside a circle (or bbox=(w, s, e, n))
    features(layer, lon=, lat=, radius_m=)  vector features near a point or in a bbox
    raster_window(layer, bbox, res)         numpy array + transform for computations in Python
"""
import math

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, box

from backend.geodata import store, vectors
from backend.geodata.catalog import RASTERS, VECTOR_LAYERS, layer
from backend.geodata.config import DERIVED

_to3310 = Transformer.from_crs(4326, 3310, always_xy=True)
_to4326 = Transformer.from_crs(3310, 4326, always_xy=True)
DEFAULT_SKIP = {"naip"}  # imagery has no single "value"; use raster_window or the map
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _cite(spec):
    return {"source": spec["source"], "date": spec.get("date"), "units": spec.get("units"), "licence": spec.get("licence")}


def _decode(spec, v):
    if v is None:
        return None
    if spec["kind"] == "categorical":
        return {"code": int(v), "label": spec["style"].get(int(v), [f"code {int(v)}"])[0]}
    x = float(v) * (spec["scale"] or 1.0)
    return round(x, 6 if abs(x) < 0.01 else 3 if abs(x) < 10 else 1)


def _available(lid):
    spec = layer(lid)
    if spec["type"] == "raster" and spec["src"]["fetch"] == "local_file":
        return (DERIVED / spec["src"]["file"]).exists()
    if spec["type"] == "vector":
        from pathlib import Path
        p = Path(spec["path"])
        return p.exists() or (lid == "buildings" and (p.parent / "buildings_ms").exists())
    return True


def _raster_ids(layers):
    ids = layers or [k for k in RASTERS if k not in DEFAULT_SKIP]
    return [k for k in ids if k in RASTERS and _available(k)]


def _vector_ids(layers):
    ids = layers or list(VECTOR_LAYERS)
    return [k for k in ids if k in VECTOR_LAYERS and _available(k)]


def point(lon, lat, layers=None, fetch=True):
    """Value of each raster layer at (lon, lat), plus what the vector layers say there."""
    x, y = _to3310.transform(lon, lat)
    out = {"lon": lon, "lat": lat, "rasters": {}, "vectors": {}, "errors": {}}
    for lid in _raster_ids(layers):
        spec = layer(lid)
        try:
            a, tr, _ = store.read(lid, (x - 1, y - 1, x + 1, y + 1), level=0, fetch=fetch)
            col, row = ~tr * (x, y)
            v = a[0, min(a.shape[1] - 1, int(row)), min(a.shape[2] - 1, int(col))]
            val = None if v == spec["nodata"] else _decode(spec, v)
            out["rasters"][lid] = {"title": spec["title"], "value": val, "res_m": spec["res"], **_cite(spec)}
        except Exception as e:  # noqa: BLE001
            out["errors"][lid] = f"{type(e).__name__}: {str(e)[:160]}"
    vids = _vector_ids(layers)
    for lid in vids:
        try:
            out["vectors"][lid] = _vector_point(lid, lon, lat)
        except Exception as e:  # noqa: BLE001
            out["errors"][lid] = f"{type(e).__name__}: {str(e)[:160]}"
    wr = windrose(lon, lat)
    if wr:
        out["fire_weather_wind_rose"] = wr
    if not out["errors"]:
        del out["errors"]
    return out


def _deg_box(lon, lat, r_m):
    dlat = r_m / 111_320
    dlon = r_m / (111_320 * max(0.2, math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _with_dist(g, lon, lat):
    g3 = g.to_crs(3310)
    p = Point(*_to3310.transform(lon, lat))
    return g.assign(distance_m=g3.distance(p).round(0).values)


def _props(row, keep=None, drop=("geometry", "bbox")):
    d = {k: v for k, v in row.items() if k not in drop and (keep is None or k in keep)}
    for k, v in list(d.items()):
        if isinstance(v, float) and math.isnan(v):
            d[k] = None
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = float(v)
    return d


def _vector_point(lid, lon, lat):
    spec = layer(lid)
    near = {"buildings": 100, "ignitions": 5000, "fire_history": 0}.get(lid, 200)
    g = vectors.query(lid, _deg_box(lon, lat, max(near, 30)))
    p = Point(lon, lat)
    res = {"title": spec["title"], **_cite(spec)}
    if spec["geom"] == "polygon" and lid != "buildings":
        inside = g[g.intersects(p)]
        res["containing"] = [_props(r) for r in inside.drop(columns="geometry").head(20).to_dict("records")]
        if near:
            g = _with_dist(g, lon, lat)
            nb = g[(g.distance_m <= near) & ~g.index.isin(inside.index)].sort_values("distance_m")
            res[f"within_{near}_m"] = [_props(r) for r in nb.drop(columns="geometry").head(10).to_dict("records")]
    elif lid == "buildings":
        g = _with_dist(g, lon, lat)
        g = g[g.distance_m <= near]
        res[f"count_within_{near}_m"] = int(len(g))
        res["nearest_m"] = float(g.distance_m.min()) if len(g) else None
    else:  # points
        g = _with_dist(g, lon, lat)
        g = g[g.distance_m <= near].sort_values("distance_m")
        res[f"count_within_{near}_m"] = int(len(g))
        res["nearest"] = [_props(r) for r in g.drop(columns="geometry").head(5).to_dict("records")]
    return res


def windrose(lon, lat):
    """Share of fire-weather days (2000-2025) with the daily mean wind from each of 8 directions."""
    p = DERIVED / "fireweather_windrose.tif"
    if not p.exists():
        return None
    import rasterio
    with rasterio.open(p) as r:
        v = next(r.sample([(lon, lat)]))
    if (v < 0).all():
        return None
    return {"from_direction_share": {s: round(float(x), 3) for s, x in zip(SECTORS, v)},
            "source": "gridMET daily wind on fire-weather days (ERC >= local 90th percentile), 2000-2025"}


def _circle_mask(tr, shape, x, y, r):
    h, w = shape
    cols = tr.c + (np.arange(w) + 0.5) * tr.a
    rows = tr.f + (np.arange(h) + 0.5) * tr.e
    return ((cols[None, :] - x) ** 2 + (rows[:, None] - y) ** 2) <= r * r


def area(lon=None, lat=None, radius_m=500, bbox=None, layers=None, res=None, fetch=True, max_features=20):
    """Summaries inside a circle (lon, lat, radius_m) or a lon/lat bbox.
    Continuous rasters: mean and percentiles of valid pixels; categorical: share of each class;
    vectors: count, nearest / largest features, and for polygons the share of the area they cover."""
    if bbox is None:
        bbox = _deg_box(lon, lat, radius_m)
        circle = True
    else:
        circle = False
    w, s, e, n = bbox
    xs, ys = _to3310.transform([w, e, w, e], [s, s, n, n])
    b3310 = (min(xs), min(ys), max(xs), max(ys))
    cx, cy = (_to3310.transform(lon, lat) if circle else ((b3310[0] + b3310[2]) / 2, (b3310[1] + b3310[3]) / 2))
    span = max(b3310[2] - b3310[0], b3310[3] - b3310[1])
    out = {"area": {"lon": lon, "lat": lat, "radius_m": radius_m} if circle else {"bbox": bbox},
           "rasters": {}, "vectors": {}, "errors": {}}
    for lid in _raster_ids(layers):
        spec = layer(lid)
        try:
            # keep reads to about 1000 x 1000 pixels: coarser level for big areas
            r_ = max(res or spec["res"], span / 1000)
            a, tr, lev = store.read(lid, b3310, res=r_, fetch=fetch)
            v = a[0]
            m = v != spec["nodata"]
            if circle:
                m &= _circle_mask(tr, v.shape, cx, cy, radius_m)
            tot = int(_circle_mask(tr, v.shape, cx, cy, radius_m).sum()) if circle else v.size
            d = {"title": spec["title"], "pixel_m": spec["res"] * 2 ** lev, "valid_share": round(m.sum() / max(1, tot), 3),
                 **_cite(spec)}
            vals = v[m]
            if vals.size:
                if spec["kind"] == "categorical":
                    codes, cnt = np.unique(vals, return_counts=True)
                    order = np.argsort(-cnt)[:10]
                    d["class_share"] = {spec["style"].get(int(codes[k]), [f"code {int(codes[k])}"])[0]:
                                        round(cnt[k] / vals.size, 3) for k in order}
                else:
                    f = vals.astype("float64") * (spec["scale"] or 1.0)
                    q = np.percentile(f, [10, 50, 90])
                    d.update(mean=_r(f.mean()), p10=_r(q[0]), median=_r(q[1]), p90=_r(q[2]), max=_r(f.max()))
            out["rasters"][lid] = d
        except Exception as e:  # noqa: BLE001
            out["errors"][lid] = f"{type(e).__name__}: {str(e)[:160]}"
    region = Point(cx, cy).buffer(radius_m) if circle else box(*b3310)
    for lid in _vector_ids(layers):
        spec = layer(lid)
        try:
            g = vectors.query(lid, bbox)
            g3 = g.to_crs(3310)
            keep = g3.intersects(region)
            g, g3 = g[keep.values], g3[keep.values]
            d = {"title": spec["title"], "count": int(len(g)), **_cite(spec)}
            if spec["geom"] == "polygon" and len(g):
                cover = g3.intersection(region).union_all().area
                d["area_share"] = round(cover / region.area, 4)
            if lid == "buildings":
                d["footprint_m2_total"] = round(float(g3.area.sum()), 0)
            elif len(g):
                gg = _with_dist(g, *_to4326.transform(cx, cy))
                sort = "acres" if "acres" in gg else "GIS_ACRES" if "GIS_ACRES" in gg else "distance_m"
                gg = gg.sort_values(sort, ascending=(sort == "distance_m"))
                d["features"] = [_props(r) for r in gg.drop(columns="geometry").head(max_features).to_dict("records")]
            out["vectors"][lid] = d
        except Exception as e:  # noqa: BLE001
            out["errors"][lid] = f"{type(e).__name__}: {str(e)[:160]}"
    if circle:
        wr = windrose(lon, lat)
        if wr:
            out["fire_weather_wind_rose"] = wr
    if not out["errors"]:
        del out["errors"]
    return out


def _r(x):
    x = float(x)
    return round(x, 6 if abs(x) < 0.01 else 3 if abs(x) < 10 else 1)


def features(layer_id, bbox=None, lon=None, lat=None, radius_m=1000, limit=200, geometry=False):
    """Vector features in a bbox or within radius_m of a point (properties; geometry as GeoJSON if asked)."""
    import json
    spec = layer(layer_id)
    if bbox is None:
        bbox = _deg_box(lon, lat, radius_m)
    g = vectors.query(layer_id, bbox)
    if lon is not None:
        g = _with_dist(g, lon, lat)
        g = g[g.distance_m <= radius_m].sort_values("distance_m")
    total = len(g)
    g = g.head(limit)
    rows = []
    for geom, r in zip(g.geometry, g.drop(columns="geometry").to_dict("records")):
        d = _props(r)
        if geometry:
            d["geometry"] = json.loads(json.dumps(geom.__geo_interface__))
        rows.append(d)
    return {"layer": layer_id, "title": spec["title"], "total": total, "returned": len(rows), "features": rows,
            **_cite(spec)}


def raster_window(layer_id, bbox, res=None, fetch=True):
    """Array for computations: bbox in lon/lat, res in metres (default native). Returns dict with the raw array
    (bands, h, w), its EPSG:3310 transform, nodata and the scale that decodes raw values."""
    w, s, e, n = bbox
    xs, ys = _to3310.transform([w, e, w, e], [s, s, n, n])
    a, tr, lev = store.read(layer_id, (min(xs), min(ys), max(xs), max(ys)), res=res, fetch=fetch)
    spec = layer(layer_id)
    return {"array": a, "transform": tr, "crs": "EPSG:3310", "nodata": spec["nodata"], "scale": spec["scale"],
            "level": lev, "pixel_m": spec["res"] * 2 ** lev, **_cite(spec)}


def fire_weather(lon, lat):
    """Fire-weather history of the gridMET cell at (lon, lat), 2000-2025: the ERC threshold, how often each month has
    fire-weather days, the wind rose on those days, and the 10 highest-ERC days with their wind and humidity."""
    import json as _json
    from datetime import date, timedelta
    dc = DERIVED / "gridmet_daily"
    if not (dc / "grid.json").exists():
        return None
    g = _json.loads((dc / "grid.json").read_text())
    la, lo = np.asarray(g["lat"]), np.asarray(g["lon"])
    r, c = int(np.argmin(np.abs(la - lat))), int(np.argmin(np.abs(lo - lon)))
    days = np.load(dc / "days_since_1900.npy")
    v = {k: np.load(dc / f"{k}.npy", mmap_mode="r")[:, r, c].astype("float32") for k in g["vars"]}
    for k in v:
        v[k][v[k] == g["nodata"]] = np.nan
        v[k] /= 10
    erc = v["erc"]
    if np.isnan(erc).all():
        return None
    p90 = float(np.nanpercentile(erc, 90))
    fwd = erc >= p90
    dates = [date(1900, 1, 1) + timedelta(days=int(d)) for d in days]
    months = np.array([d.month for d in dates])
    years = g["years"][1] - g["years"][0] + 1
    by_month = {m: round(float(fwd[months == m].sum()) / years, 1) for m in range(1, 13)}
    top = np.argsort(-np.nan_to_num(erc, nan=-1))[:10]
    return {
        "cell": {"lat": float(la[r]), "lon": float(lo[c])},
        "years": g["years"], "erc_p90_threshold": round(p90, 1),
        "fire_weather_days_per_year": round(float(fwd.sum()) / years, 1),
        "fire_weather_days_per_year_by_month": by_month,
        "wind_rose_on_fire_weather_days": (windrose(lon, lat) or {}).get("from_direction_share"),
        "top_erc_days": [{"date": dates[k].isoformat(), "erc": _r(erc[k]), "wind_mean_m_s": _r(v["vs"][k]),
                          "wind_from_deg": _r(v["th"][k]), "rh_min_pct": _r(v["rmin"][k]),
                          "fm100_pct": _r(v["fm100"][k])} for k in top],
        "source": "gridMET daily (Abatzoglou 2013), 4 km; ERC fuel model G; winds are daily means at 10 m",
    }
