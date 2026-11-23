"""
Hourly NOAA HRRR weather (past 24 h and the forecast ahead) on fixed map tiles, for the time slider on the risk map.

A tile is TILE x TILE cells of a lat/lon lattice with spacing `step` degrees; tile (ti, tj) covers
lat [ti*TILE*step, (ti+1)*TILE*step] and lon likewise, and carries its (TILE+1)^2 corner points so the browser can
interpolate inside it. Because tiles are fixed, panning the map only asks for tiles it has not seen.

Values come from Open-Meteo (models=gfs_hrrr), which serves HRRR's 3 km output. Each lattice point is cached in a
small SQLite file (backend/cache/) so a restart or a slow upstream does not mean refetching; if Open-Meteo does not
answer, the last good values (up to 12 h old) are served and marked stale. Nothing is filled in or invented.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend import live_sources as live

TILE = 6
STEPS = [0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0]  # 0.03 deg ~ the 3 km HRRR grid
PAST_H, AHEAD_H = 24, 48
FRESH_S = 45 * 60          # HRRR runs hourly
STALE_MAX_S = 12 * 3600
VARS = {  # key -> (Open-Meteo hourly variable, scale applied before rounding to an integer)
    "temp_f": ("temperature_2m", 1),
    "rh": ("relative_humidity_2m", 1),
    "wspd_mph10": ("wind_speed_10m", 10),
    "wdir": ("wind_direction_10m", 1),
    "gust_mph10": ("wind_gusts_10m", 10),
    "precip_in1000": ("precipitation", 1000),
    "cloud": ("cloud_cover", 1),
}
DB_PATH = Path(__file__).resolve().parent / "cache" / "hrrr_points.sqlite"
_db_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS pts (k TEXT PRIMARY KEY, fetched REAL, t0 INTEGER, data TEXT)")
        _conn.execute("DELETE FROM pts WHERE fetched < ?", (time.time() - 3 * 86400,))
        _conn.commit()
    return _conn


def _key(lat: float, lon: float) -> str:
    return f"{lat:.4f},{lon:.4f}"


def _load(keys: List[str]) -> Dict[str, Tuple[float, int, dict]]:
    with _db_lock:
        db = _db()
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT k, fetched, t0, data FROM pts WHERE k IN ({','.join('?' * len(part))})"
            for k, f, t0, d in db.execute(q, part):
                out[k] = (f, t0, json.loads(d))
        return out


def _save(rows: List[Tuple[str, float, int, dict]]) -> None:
    with _db_lock:
        db = _db()
        db.executemany("INSERT OR REPLACE INTO pts VALUES (?,?,?,?)", [(k, f, t0, json.dumps(d, separators=(",", ":"))) for k, f, t0, d in rows])
        db.commit()


def _epoch_hour(iso: str) -> int:
    import datetime as dt
    return int(dt.datetime.fromisoformat(iso + "+00:00").timestamp() // 3600)


def _fetch(points: List[Tuple[float, float]]) -> Dict[str, Tuple[int, dict]]:
    params = {"latitude": ",".join(f"{a:.4f}" for a, _ in points), "longitude": ",".join(f"{o:.4f}" for _, o in points),
              "models": "gfs_hrrr", "hourly": ",".join(v[0] for v in VARS.values()), "past_hours": PAST_H,
              "forecast_hours": AHEAD_H, "timezone": "UTC", "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
              "precipitation_unit": "inch"}
    d = live.om_get(params, len(points), timeout=12).json()
    d = d if isinstance(d, list) else [d]
    out = {}
    for (lat, lon), r in zip(points, d):
        h = r["hourly"]
        vals = {k: [None if x is None else round(x * s) for x in h[name]] for k, (name, s) in VARS.items()}
        out[_key(lat, lon)] = (_epoch_hour(h["time"][0]), vals)
    return out


def pick_step(span_deg: float, cols: int = 12) -> float:
    return next((s for s in STEPS if span_deg / s <= cols), STEPS[-1])


def forecast_tile(step: float, ti: int, tj: int) -> dict:
    if step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}")
    n = TILE + 1
    lats = [round((ti * TILE + a) * step, 4) for a in range(n)]
    lons = [round((tj * TILE + b) * step, 4) for b in range(n)]
    pts = [(la, lo) for la in lats for lo in lons]
    inside = [live._in_hrrr(la, lo) for la, lo in pts]
    keys = [_key(*p) for p in pts]
    now = time.time()
    have = _load([k for k, ok in zip(keys, inside) if ok])
    need = [p for p, k, ok in zip(pts, keys, inside) if ok and (k not in have or now - have[k][0] > FRESH_S)]
    err = None
    if need:
        try:
            got = _fetch(need)
            _save([(k, now, t0, v) for k, (t0, v) in got.items()])
            for k, (t0, v) in got.items():
                have[k] = (now, t0, v)
            live.STATUS["noaa_hrrr_forecast"] = {"ok": True, "checked_at": live._now_iso(), "error": None}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:160]}"
            live.STATUS["noaa_hrrr_forecast"] = {"ok": False, "checked_at": live._now_iso(), "error": err}
    # Common hour axis for the tile: PAST_H before this hour to AHEAD_H after it
    h_now = int(now // 3600)
    t0, H = h_now - PAST_H, PAST_H + AHEAD_H
    out = {k: [None] * (H * n * n) for k in VARS}
    oldest, stale_pts, missing_pts, h_first, h_last = None, 0, 0, None, None
    for idx, (k, ok) in enumerate(zip(keys, inside)):
        if not ok:
            continue
        rec = have.get(k)
        if not rec or now - rec[0] > STALE_MAX_S:
            missing_pts += 1
            continue
        fetched, pt0, vals = rec
        if now - fetched > FRESH_S:
            stale_pts += 1
        oldest = fetched if oldest is None else min(oldest, fetched)
        for var, arr in vals.items():
            if var not in out:
                continue
            for j, x in enumerate(arr):
                h = pt0 + j - t0
                if 0 <= h < H and x is not None:
                    out[var][h * n * n + idx] = x
                    if var == "temp_f":
                        h_first = h if h_first is None else min(h_first, h)
                        h_last = h if h_last is None else max(h_last, h)
    if missing_pts and missing_pts == sum(inside) and err:
        raise RuntimeError(f"No weather cached for this area yet: {err}")
    return {"step": step, "tile": TILE, "ti": ti, "tj": tj, "lat0": lats[0], "lon0": lons[0], "n": n,
            "t0": t0, "hours": H, "now_hour": h_now,
            "first_hour": None if h_first is None else t0 + h_first, "last_hour": None if h_last is None else t0 + h_last,
            "vars": out,
            "scale": {"temp_f": 1, "rh": 1, "wspd_mph10": 10, "wdir": 1, "gust_mph10": 10, "precip_in1000": 1000, "cloud": 1},
            "fetched_utc": live._iso(oldest) if oldest else None, "stale_points": stale_pts, "missing_points": missing_pts,
            "upstream_error": err,
            "source": "NOAA HRRR 3 km (hourly; past hours stitched from earlier runs) via Open-Meteo, models=gfs_hrrr"}


def tile_index(lat: float, lon: float, step: float) -> Tuple[int, int]:
    span = TILE * step
    return math.floor(lat / span), math.floor(lon / span)
