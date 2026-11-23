"""
Hourly weather on fixed map tiles for the time slider on the risk map: the past 24 h and about 10 days ahead.

Data come from backend/noaa_grid.py, which reads NOAA HRRR (3 km, to 48 h) and the National Blend of Models (NBM,
2.5 km, beyond that, 3- to 6-hourly) straight from NOAA's open-data buckets on AWS onto a 0.03 degree grid over
California. No per-request upstream calls happen here: a tile is read from that local grid.

A tile is TILE x TILE cells of a lat/lon lattice with spacing `step` degrees; tile (ti, tj) covers
lat [ti*TILE*step, (ti+1)*TILE*step] and lon likewise, and carries its (TILE+1)^2 corner points so the browser can
interpolate inside it. Because tiles are fixed, panning the map only asks for tiles it has not seen.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from backend import noaa_grid as ng

TILE = 6
STEPS = [0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0]
_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _weights(lats: np.ndarray, lons: np.ndarray):
    """Bilinear corner indices and weights into the stored grid for points; `ok` marks points inside it."""
    fy, fx = (lats - ng.LAT0) / ng.D, (lons - ng.LON0) / ng.D
    ok = (fy >= 0) & (fx >= 0) & (fy <= ng.NY - 1) & (fx <= ng.NX - 1)
    j0 = np.clip(np.floor(fy).astype(int), 0, ng.NY - 2)
    i0 = np.clip(np.floor(fx).astype(int), 0, ng.NX - 2)
    wy, wx = np.clip(fy - j0, 0, 1), np.clip(fx - i0, 0, 1)
    return j0, i0, wy, wx, ok


def _interp(a: np.ndarray, w) -> np.ndarray:
    """(fields, points) float array from an int16 (fields, NY, NX) hour; NaN where a corner is missing."""
    j0, i0, wy, wx, ok = w
    c = [a[:, j0, i0], a[:, j0, i0 + 1], a[:, j0 + 1, i0], a[:, j0 + 1, i0 + 1]]
    c = [np.where(x == ng.MISSING, np.nan, x.astype(np.float32)) for x in c]
    v = c[0] * (1 - wx) * (1 - wy) + c[1] * wx * (1 - wy) + c[2] * (1 - wx) * wy + c[3] * wx * wy
    v[:, ~ok] = np.nan
    return v


def _hours() -> Tuple[list, dict, int]:
    cat, ver = ng.catalog()
    now_h = int(time.time() // 3600)
    hrs = sorted(v for v in cat if v >= now_h - ng.PAST_H)
    return hrs, cat, ver


def forecast_tile(step: float, ti: int, tj: int) -> dict:
    if step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}")
    hrs, cat, ver = _hours()
    if not hrs:
        st = ng.status()
        raise RuntimeError("The weather grid is still being downloaded from NOAA (first start takes a few minutes)"
                           + (f"; last error: {st['error']}" if st.get("error") else ""))
    now_h = int(time.time() // 3600)
    ck = (step, ti, tj, ver, now_h)
    if ck in _cache:
        _cache.move_to_end(ck)
        return _cache[ck]
    n = TILE + 1
    lats = np.array([round((ti * TILE + a) * step, 4) for a in range(n)])
    lons = np.array([round((tj * TILE + b) * step, 4) for b in range(n)])
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    w = _weights(LA.ravel(), LO.ravel())
    outside = int((~w[4]).sum())
    vals = {f: [] for f in ng.FIELDS}
    times, src = [], []
    if outside < n * n:
        for v in hrs:
            a = ng.hour_array(v)
            if a is None:
                continue
            x = _interp(a, w)
            for k, f in enumerate(ng.FIELDS):
                vals[f].extend(None if math.isnan(y) else int(round(y)) for y in x[k])
            times.append(v)
            src.append("h" if cat[v]["product"] == "hrrr" else "n")
    runs = {}
    for v in times:
        m = cat[v]
        if v >= now_h:
            runs[m["product"]] = max(runs.get(m["product"], 0), m["run"])
    out = {"step": step, "tile": TILE, "ti": ti, "tj": tj, "lat0": float(lats[0]), "lon0": float(lons[0]), "n": n,
           "times": times, "src": "".join(src), "now_hour": now_h, "vars": vals, "scale": ng.SCALE,
           "runs_utc": {k: ng._iso(r) for k, r in runs.items()}, "outside_points": outside,
           "source": "NOAA HRRR 3 km (past 24 h and next 48 h, hourly) and NOAA National Blend of Models 2.5 km "
                     "(after that, 3- to 6-hourly), read from NOAA's open data on AWS"}
    _cache[ck] = out
    while len(_cache) > 200:
        _cache.popitem(last=False)
    return out


def point_now(lat: float, lon: float) -> Optional[dict]:
    """Model weather at a point for the current hour (for the Conditions panel), or None outside the grid."""
    hrs, cat, _ = _hours()
    now_h = int(time.time() // 3600)
    if now_h not in cat:
        return None
    w = _weights(np.array([lat]), np.array([lon]))
    if not w[4][0]:
        return None
    a = ng.hour_array(now_h)
    if a is None:
        return None
    x = _interp(a, w)[:, 0]
    if np.isnan(x[:5]).any():
        return None
    f = dict(zip(ng.FIELDS, (float(y) / ng.SCALE[k] for k, y in zip(ng.FIELDS, x))))
    spd = math.hypot(f["u_mph10"], f["v_mph10"])
    wdir = (math.degrees(math.atan2(-f["u_mph10"], -f["v_mph10"])) + 360) % 360
    m = cat[now_h]
    from backend.live_sources import cardinal
    return {"source": f"NOAA HRRR 3 km, run {ng._iso(m['run'])} (+{m['lead']} h), from NOAA open data on AWS",
            "model_run_utc": ng._iso(m["run"]), "valid_time_utc": ng._iso(now_h), "grid_lat": round(lat, 3), "grid_lon": round(lon, 3),
            "temperature_f": round(f["temp_f10"], 1), "relative_humidity_pct": round(f["rh"]),
            "wind_speed_mph": round(spd, 1), "wind_gust_mph": round(f["gust_mph10"], 1),
            "wind_direction_deg": round(wdir), "wind_direction_cardinal": cardinal(wdir),
            "precip_in_per_h": round(f["precip_in1000"], 3) if not math.isnan(f["precip_in1000"]) else None,
            "cloud_pct": None if math.isnan(f["cloud"]) else round(f["cloud"]),
            "fetched_utc": dt.datetime.fromtimestamp(m["fetched"], dt.timezone.utc).isoformat(timespec="minutes") if m.get("fetched") else None}


def tile_index(lat: float, lon: float, step: float) -> Tuple[int, int]:
    span = TILE * step
    return math.floor(lat / span), math.floor(lon / span)
