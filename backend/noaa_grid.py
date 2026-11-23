"""
NOAA weather models read straight from NOAA's open-data buckets on AWS, kept on a California lat/lon grid.

  HRRR (noaa-hrrr-bdp-pds): 3 km, hourly runs. Past hours use the 1-hour forecast of the run that started an hour
       earlier (the closest the model gets to what happened); the next 48 h use the newest complete run that reaches
       that far (every run reaches 18 h; the 00/06/12/18 UTC runs reach 48 h).
  NBM  (noaa-nbm-grib2-pds): National Blend of Models, 2.5 km. Picks up where HRRR stops and runs to about 11 days,
       from the 00/06/12/18 UTC runs (hourly to 36 h, then 3-hourly, then 6-hourly).

Each GRIB2 file is 100-170 MB, but NOAA publishes an .idx next to it with the byte offset of every field, so only the
seven fields used here are downloaded (about 15 MB per HRRR hour), using HTTP range requests. Fields are decoded with
ecCodes, moved from the model's Lambert conformal grid to a regular 0.03 degree lat/lon grid over California
(bilinear), converted to US units and stored per valid hour as int16 under backend/cache/noaa/. Values are what the
models say; nothing is filled in. Hours a download has not reached are simply absent.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

log = logging.getLogger("noaa_grid")

# California and its edges (the area the risk map covers). 0.03 degrees ~ the 3 km HRRR grid.
LAT0, LON0, D = 31.5, -126.0, 0.03
NY, NX = 384, 434  # to 42.99 N, -113.01 E
FIELDS = ["temp_f10", "rh", "u_mph10", "v_mph10", "gust_mph10", "precip_in1000", "cloud"]  # stored order
SCALE = {"temp_f10": 10, "rh": 1, "u_mph10": 10, "v_mph10": 10, "gust_mph10": 10, "precip_in1000": 1000, "cloud": 1}
MISSING = -32768
PAST_H = 24
CACHE = Path(__file__).resolve().parent / "cache" / "noaa"
HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
NBM_URL = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
# Use an hourly (non-synoptic) HRRR run for the next 18 h only every N hours; each refresh is ~18 files x 15 MB.
HRRR_REFRESH_H = int(os.environ.get("WX_HRRR_REFRESH_H", "3"))
MPS_TO_MPH = 2.2369363

_http = requests.Session()
_http.headers["User-Agent"] = "wildfire-ai-map/1.0"
_lock = threading.Lock()
_status: Dict = {"state": "idle", "last_cycle_utc": None, "error": None, "downloaded_mb": 0.0, "hours": 0}


def _hr(h: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(h * 3600, dt.timezone.utc)


def _iso(h: int) -> str:
    return _hr(h).strftime("%Y-%m-%dT%H:00Z")


# ------------------------------------------------------------------------------------------------------------ sources
def _hrrr_path(run: int, lead: int) -> str:
    t = _hr(run)
    return f"{HRRR_URL}/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf{lead:02d}.grib2"


def _nbm_path(run: int, lead: int) -> str:
    t = _hr(run)
    return f"{NBM_URL}/blend.{t:%Y%m%d}/{t:%H}/core/blend.t{t:%H}z.core.f{lead:03d}.co.grib2"


def _hrrr_records(lead: int) -> Dict[str, str]:
    """idx record (variable:level:time:) for each field we need from an HRRR file."""
    return {"TMP": f"TMP:2 m above ground:{lead} hour fcst:", "RH": f"RH:2 m above ground:{lead} hour fcst:",
            "U": f"UGRD:10 m above ground:{lead} hour fcst:", "V": f"VGRD:10 m above ground:{lead} hour fcst:",
            "GUST": f"GUST:surface:{lead} hour fcst:", "APCP": f"APCP:surface:{lead - 1}-{lead} hour acc fcst:",
            "TCDC": f"TCDC:entire atmosphere:{lead} hour fcst:"}


def _nbm_records(lead: int) -> Dict[str, str]:
    return {"TMP": f"TMP:2 m above ground:{lead} hour fcst:", "RH": f"RH:2 m above ground:{lead} hour fcst:",
            "WIND": f"WIND:10 m above ground:{lead} hour fcst:", "WDIR": f"WDIR:10 m above ground:{lead} hour fcst:",
            "GUST": f"GUST:10 m above ground:{lead} hour fcst:", "APCP": f"APCP:surface:{lead - 1}-{lead} hour acc fcst:",
            "TCDC": f"TCDC:surface:{lead} hour fcst:"}


def _get(url: str, headers: Optional[dict] = None, timeout: float = 30, tries: int = 3) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = _http.get(url, headers=headers, timeout=timeout)
            if r.status_code in (200, 206, 404, 403):
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except requests.RequestException as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url.rsplit('/', 1)[-1]}: {last}")


def _exists(url: str) -> bool:
    return _get(url + ".idx", timeout=15).status_code == 200


def _ranges(url: str, records: Dict[str, str]) -> Dict[str, Tuple[int, Optional[int]]]:
    r = _get(url + ".idx")
    if r.status_code != 200:
        raise FileNotFoundError(url)
    lines = [ln.split(":", 3) for ln in r.text.strip().splitlines()]
    offs = [int(p[1]) for p in lines]
    out = {}
    for name, rec in records.items():
        for i, p in enumerate(lines):
            if len(p) == 4 and p[3] == rec:
                out[name] = (offs[i], offs[i + 1] - 1 if i + 1 < len(offs) else None)
                break
    return out


# --------------------------------------------------------------------------------------------------- GRIB + regrid
_regrid_cache: Dict[tuple, tuple] = {}


def _decode(buf: bytes):
    import eccodes
    h = eccodes.codes_new_from_message(buf)
    try:
        eccodes.codes_set(h, "missingValue", 1e20)
        g = {k: eccodes.codes_get(h, k) for k in ("gridType", "Nx", "Ny", "LoVInDegrees", "Latin1InDegrees",
                                                   "Latin2InDegrees", "LaDInDegrees", "DxInMetres", "DyInMetres",
                                                   "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
                                                   "jScansPositively", "iScansNegatively")}
        try:
            g["radius"] = float(eccodes.codes_get(h, "radius"))
        except Exception:  # noqa: BLE001
            g["radius"] = 6371229.0
        g["uv_grid"] = int(eccodes.codes_get(h, "uvRelativeToGrid"))
        vals = eccodes.codes_get_values(h).astype(np.float32)
    finally:
        eccodes.codes_release(h)
    vals[vals > 1e19] = np.nan
    if g["gridType"] != "lambert" or g["iScansNegatively"] or not g["jScansPositively"]:
        raise ValueError(f"unexpected grid {g['gridType']} scan {g['iScansNegatively']}/{g['jScansPositively']}")
    return vals.reshape(g["Ny"], g["Nx"]), g


def _regridder(g: dict):
    """Bilinear weights from the model's Lambert grid to our lat/lon grid, and the wind rotation angle there."""
    sig = tuple(round(float(g[k]), 6) for k in ("Nx", "Ny", "LoVInDegrees", "Latin1InDegrees", "Latin2InDegrees",
                                                 "DxInMetres", "latitudeOfFirstGridPointInDegrees",
                                                 "longitudeOfFirstGridPointInDegrees", "radius"))
    if sig in _regrid_cache:
        return _regrid_cache[sig]
    import pyproj
    proj = pyproj.Proj(f"+proj=lcc +lat_1={g['Latin1InDegrees']} +lat_2={g['Latin2InDegrees']} +lat_0={g['LaDInDegrees']} "
                       f"+lon_0={g['LoVInDegrees']} +a={g['radius']} +b={g['radius']} +units=m")
    lon1 = g["longitudeOfFirstGridPointInDegrees"]
    x0, y0 = proj(lon1 - 360 if lon1 > 180 else lon1, g["latitudeOfFirstGridPointInDegrees"])
    lat = LAT0 + D * np.arange(NY)
    lon = LON0 + D * np.arange(NX)
    LON, LAT = np.meshgrid(lon, lat)
    x, y = proj(LON, LAT)
    fx, fy = (x - x0) / g["DxInMetres"], (y - y0) / g["DyInMetres"]
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ok = (i0 >= 0) & (j0 >= 0) & (i0 < g["Nx"] - 1) & (j0 < g["Ny"] - 1)
    i0, j0 = np.clip(i0, 0, g["Nx"] - 2), np.clip(j0, 0, g["Ny"] - 2)
    wx, wy = (fx - i0).astype(np.float32), (fy - j0).astype(np.float32)
    # Lambert conformal: grid north differs from true north by n * (lon - LoV); n = sin(latin) for a tangent cone
    l1, l2 = math.radians(g["Latin1InDegrees"]), math.radians(g["Latin2InDegrees"])
    n = math.sin(l1) if abs(l1 - l2) < 1e-9 else (math.log(math.cos(l1) / math.cos(l2)) /
                                                   math.log(math.tan(math.pi / 4 + l2 / 2) / math.tan(math.pi / 4 + l1 / 2)))
    lov = g["LoVInDegrees"] - 360 if g["LoVInDegrees"] > 180 else g["LoVInDegrees"]
    ang = (n * np.radians(LON - lov)).astype(np.float32)
    _regrid_cache[sig] = (i0, j0, wx, wy, ok, ang)
    return _regrid_cache[sig]


def _sample(field: np.ndarray, rg) -> np.ndarray:
    i0, j0, wx, wy, ok, _ = rg
    v = (field[j0, i0] * (1 - wx) * (1 - wy) + field[j0, i0 + 1] * wx * (1 - wy)
         + field[j0 + 1, i0] * (1 - wx) * wy + field[j0 + 1, i0 + 1] * wx * wy)
    v[~ok] = np.nan
    return v


def _download_fields(url: str, records: Dict[str, str]) -> Tuple[Dict[str, np.ndarray], dict, float]:
    rng = _ranges(url, records)
    missing = set(records) - set(rng)
    if missing:
        raise KeyError(f"{url.rsplit('/', 1)[-1]} lacks {sorted(missing)}")
    out, grid, nbytes = {}, None, 0
    for name, (a, b) in rng.items():
        r = _get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"}, timeout=60)
        if r.status_code != 206:
            raise RuntimeError(f"range request returned HTTP {r.status_code}")
        nbytes += len(r.content)
        vals, g = _decode(r.content)
        grid = grid or g
        out[name] = (vals, g)
    return out, grid, nbytes / 1e6


def _to_store(arrs: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.full((len(FIELDS), NY, NX), MISSING, dtype=np.int16)
    for k, name in enumerate(FIELDS):
        a = arrs[name] * SCALE[name]
        good = np.isfinite(a)
        out[k][good] = np.clip(np.round(a[good]), -32767, 32767).astype(np.int16)
    return out


def fetch_hour(product: str, run: int, lead: int) -> Tuple[np.ndarray, float]:
    """Download, decode and regrid one forecast hour. Returns the (fields, NY, NX) int16 array and MB downloaded."""
    if product == "hrrr":
        f, g, mb = _download_fields(_hrrr_path(run, lead), _hrrr_records(lead))
        rg = _regridder(g)
        u, v = _sample(f["U"][0], rg), _sample(f["V"][0], rg)
        if f["U"][1]["uv_grid"]:  # grid-relative winds -> east/north
            c, s = np.cos(rg[5]), np.sin(rg[5])
            u, v = c * u + s * v, -s * u + c * v
        gust = _sample(f["GUST"][0], rg)
    else:
        f, g, mb = _download_fields(_nbm_path(run, lead), _nbm_records(lead))
        rg = _regridder(g)
        spd = _sample(f["WIND"][0], rg)
        # Direction is not interpolated as an angle: go through the components on the source grid
        wd = np.radians(f["WDIR"][0])
        su, sv = -f["WIND"][0] * np.sin(wd), -f["WIND"][0] * np.cos(wd)
        u, v = _sample(su, rg), _sample(sv, rg)
        if f["WDIR"][1]["uv_grid"]:
            c, s = np.cos(rg[5]), np.sin(rg[5])
            u, v = c * u + s * v, -s * u + c * v
        k = np.where(np.hypot(u, v) > 0.01, spd / np.maximum(np.hypot(u, v), 0.01), 1.0)  # keep the speed field's speed
        u, v = u * k, v * k
        gust = _sample(f["GUST"][0], rg)
    arrs = {
        "temp_f10": (_sample(f["TMP"][0], rg) - 273.15) * 9 / 5 + 32,
        "rh": np.clip(_sample(f["RH"][0], rg), 0, 100),
        "u_mph10": u * MPS_TO_MPH, "v_mph10": v * MPS_TO_MPH, "gust_mph10": gust * MPS_TO_MPH,
        "precip_in1000": np.clip(_sample(f["APCP"][0], rg), 0, None) / 25.4,  # kg/m2 = mm in the last hour -> in/h
        "cloud": np.clip(_sample(f["TCDC"][0], rg), 0, 100),
    }
    return _to_store(arrs), mb


# ---------------------------------------------------------------------------------------------------------- catalog
def _meta_path(v: int) -> Path:
    return CACHE / f"{v}.json"


def _data_path(v: int) -> Path:
    return CACHE / f"{v}.npy"


_catalog: Dict[int, dict] = {}
_catalog_version = 0
_mmaps: Dict[int, Tuple[float, np.ndarray]] = {}


def _load_catalog() -> None:
    global _catalog_version
    CACHE.mkdir(parents=True, exist_ok=True)
    cat = {}
    for p in CACHE.glob("*.json"):
        try:
            m = json.loads(p.read_text())
            if _data_path(int(p.stem)).exists():
                cat[int(p.stem)] = m
        except Exception:  # noqa: BLE001
            continue
    with _lock:
        _catalog.clear()
        _catalog.update(cat)
        _catalog_version += 1


def _store(v: int, arr: np.ndarray, meta: dict) -> None:
    global _catalog_version
    tmp = CACHE / f".{v}.npy.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, _data_path(v))
    _meta_path(v).write_text(json.dumps(meta))
    with _lock:
        _catalog[v] = meta
        _mmaps.pop(v, None)
        _catalog_version += 1


def catalog() -> Tuple[Dict[int, dict], int]:
    with _lock:
        return dict(_catalog), _catalog_version


def hour_array(v: int) -> Optional[np.ndarray]:
    p = _data_path(v)
    try:
        mt = p.stat().st_mtime
    except FileNotFoundError:
        return None
    with _lock:
        hit = _mmaps.get(v)
        if hit and hit[0] == mt:
            return hit[1]
    a = np.load(p, mmap_mode="r")
    with _lock:
        _mmaps[v] = (mt, a)
    return a


# --------------------------------------------------------------------------------------------------------- planning
def _latest_complete(now_h: int, path, last_lead, step: int, back: int) -> Optional[int]:
    """Newest run (a multiple of `step` hours) whose last forecast hour is already in the bucket."""
    r = now_h - now_h % step
    for _ in range(back):
        if _exists(path(r, last_lead(r))):
            return r
        r -= step
    return None


def _nbm_leads(run: int) -> List[int]:
    t = _hr(run)
    prefix = f"blend.{t:%Y%m%d}/{t:%H}/core/"
    leads, tok = [], None
    while True:
        p = {"list-type": 2, "prefix": prefix}
        if tok:
            p["continuation-token"] = tok
        txt = _get(NBM_URL + "/?" + "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in p.items())).text
        leads += [int(x) for x in re.findall(r"\.f(\d{3})\.co\.grib2</Key>", txt)]
        m = re.search(r"<NextContinuationToken>([^<]*)", txt)
        if not m:
            break
        tok = m.group(1)
    return sorted(set(leads))


def plan(now_h: int) -> Dict[int, Tuple[str, int, int]]:
    """valid hour -> (product, run, lead): the newest model run that covers each hour."""
    want: Dict[int, Tuple[str, int, int]] = {}
    r_any = _latest_complete(now_h, _hrrr_path, lambda r: 18, 1, 6)
    r_syn = _latest_complete(now_h, _hrrr_path, lambda r: 48, 6, 4)
    if r_any is None:
        return want
    # forecasts beyond the newest run's first hour: the newest run made on the refresh schedule
    r_fc = max([r for r in range(r_any - HRRR_REFRESH_H + 1, r_any + 1) if r % HRRR_REFRESH_H == 0] + [r_syn or -1])
    for v in range(now_h - PAST_H, r_any + 2):
        want[v] = ("hrrr", v - 1, 1)
    for v in range(r_any + 2, r_fc + 19):
        want.setdefault(v, ("hrrr", r_fc, v - r_fc))
    hrrr_end = r_fc + 18
    if r_syn is not None and r_syn + 48 > hrrr_end:
        for v in range(hrrr_end + 1, r_syn + 49):
            want[v] = ("hrrr", r_syn, v - r_syn)
        hrrr_end = r_syn + 48
    n_syn = _latest_complete(now_h, _nbm_path, lambda r: 264, 6, 4)
    if n_syn is not None:
        for lead in _nbm_leads(n_syn):
            v = n_syn + lead
            # past 36 h NBM's files hold every field only at 3-hourly / 6-hourly steps; others carry rain alone
            if v > hrrr_end and (lead <= 36 or (lead <= 192 and lead % 3 == 0) or lead % 6 == 0):
                want[v] = ("nbm", n_syn, lead)
    return want


def cycle(now_h: Optional[int] = None, workers: int = 6) -> dict:
    """Bring the cache up to date: download every planned hour that is missing or from an older run."""
    now_h = now_h if now_h is not None else int(time.time() // 3600)
    _load_catalog()
    want = plan(now_h)
    cat, _ = catalog()
    todo = [(v, p) for v, p in want.items()
            if not (v in cat and (cat[v]["product"], cat[v]["run"], cat[v]["lead"]) == (p[0], p[1], p[2]))]
    # nearest hours first: now and the next day, then further ahead, then the past
    todo.sort(key=lambda x: (0 if 0 <= x[0] - now_h <= 48 else 1 if x[0] > now_h else 2, abs(x[0] - now_h)))
    done, failed, mb = 0, [], 0.0

    def job(item):
        v, (prod, run, lead) = item
        arr, m = fetch_hour(prod, run, lead)
        _store(v, arr, {"product": prod, "run": run, "lead": lead, "valid": v, "fetched": time.time()})
        return m

    _status.update(state="downloading", todo=len(todo))
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(job, it): it for it in todo}
        for fu, it in futs.items():
            try:
                mb += fu.result()
                done += 1
            except Exception as e:  # noqa: BLE001
                failed.append(f"{it[1][0]} {_iso(it[1][1])}+{it[1][2]}h: {type(e).__name__}: {str(e)[:120]}")
            _status["todo"] = len(todo) - done - len(failed)
    # Drop hours that fell out of the window, and hours a newer run no longer has (e.g. NBM's 3-hourly steps ending
    # at a different time), so runs are not mixed. A failed download keeps the older value until it is replaced.
    planned_products = {p[0] for p in want.values()}
    for v in list(cat):
        stray = v not in want and cat[v]["product"] in planned_products and v > now_h
        if v < now_h - PAST_H - 1 or v > now_h + 270 or stray:
            for p in (_data_path(v), _meta_path(v)):
                p.unlink(missing_ok=True)
    _load_catalog()
    cat, _ = catalog()
    _status.update(state="idle", last_cycle_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   error=failed[0] if failed else None, failed=len(failed), downloaded_mb=round(mb, 1),
                   hours=len(cat), planned=len(want))
    return {"planned": len(want), "downloaded": done, "failed": failed, "mb": round(mb, 1)}


def status() -> dict:
    cat, _ = catalog()
    now_h = int(time.time() // 3600)
    runs = {}
    for m in cat.values():
        if m["valid"] >= now_h:
            runs[m["product"]] = max(runs.get(m["product"], 0), m["run"])
    return {**_status, "hours": len(cat), "latest_runs_utc": {k: _iso(v) for k, v in runs.items()}}


_thread: Optional[threading.Thread] = None


def start_background(period_s: int = 600) -> None:
    """Keep the cache fresh from inside the API server (one thread; new runs appear about hourly)."""
    global _thread
    if _thread or os.environ.get("WX_INGEST", "1") == "0":
        return
    _load_catalog()

    def loop():
        while True:
            try:
                r = cycle()
                log.info("noaa cycle: %s", {k: v for k, v in r.items() if k != "failed"})
            except Exception as e:  # noqa: BLE001
                _status.update(state="error", error=f"{type(e).__name__}: {str(e)[:200]}")
                log.exception("noaa cycle failed")
            time.sleep(period_s)

    _thread = threading.Thread(target=loop, name="noaa-ingest", daemon=True)
    _thread.start()


if __name__ == "__main__":  # python -m backend.noaa_grid  -> one update cycle
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(cycle(), indent=1))
