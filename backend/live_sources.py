"""
Live, real data sources for the map view. Every function fetches from a public service (or a real local dataset),
caches the result briefly, and records its status so the UI can show where each layer came from and when.

Sources (all public, no API key):
  NIFC WFIGS current perimeters ............ services3.arcgis.com/T4QMspbfLg3qTGWY (WFIGS_Interagency_Perimeters_Current)
  CAL FIRE historical perimeters (1950+) ... services1.arcgis.com/jUJYIo9tSA7EHvfZ (California_Historic_Fire_Perimeters)
  CAL FIRE active incidents ................ incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList
  NASA FIRMS VIIRS 375 m (S-NPP, NOAA-20, NOAA-21), last 24 h ... firms.modaps.eosdis.nasa.gov/data/active_fire/
  CDEC RAWS stations, sensor 11 (10-h fuel stick moisture) ... cdec.water.ca.gov
  NOAA HRRR via Open-Meteo (current conditions, wind grid) ... api.open-meteo.com (models=gfs_hrrr)
  NWS active alerts (Red Flag Warnings) .... api.weather.gov
  USGS 3DEP elevation (slope computed here) ... elevation.nationalmap.gov 3DEPElevation ImageServer
  Sentinel-2 NDVI (Esri Sentinel-2 Views, one dated scene) ... sentinel.arcgis.com Sentinel2 ImageServer
  CAL FIRE FHSZ: SRA 2023 (FHSZSRA_23_3) and LRA 2025 (FHSZLRA25_v1_All) ... services1.arcgis.com/jUJYIo9tSA7EHvfZ
  USDA Wildfire Risk to Communities annual burn probability (2024 release) ... imagery.geoplatform.gov ImageServer
  ESA WorldCover 2021 10 m land cover (point lookup) ... esa-worldcover.s3.eu-central-1.amazonaws.com
  Microsoft US Building Footprints (local extract; scripts/data/extract_ms_buildings_region.py)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "backend" / "static"
BUILDINGS_PARQUET = BASE_DIR / "data" / "ms_buildings_bayarea.parquet"
HEADERS = {"User-Agent": "wildfire-ai-research-demo/1.0 (academic research prototype)"}
CA_BBOX = (-124.5, 32.4, -114.1, 42.1)  # min_lon, min_lat, max_lon, max_lat

Bbox = Tuple[float, float, float, float]

# ----------------------------------------------------------------------------- caching and status
_cache: Dict[str, Tuple[float, Any]] = {}
_lock = threading.Lock()
STATUS: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def cached(source: str, ttl: int) -> Callable:
    """Cache by arguments for `ttl` seconds and record success/failure for `source`."""
    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = source + json.dumps([args, kwargs], default=str, sort_keys=True)
            with _lock:
                hit = _cache.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
            try:
                val = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                STATUS[source] = {"ok": False, "checked_at": _now_iso(), "error": f"{type(e).__name__}: {str(e)[:200]}"}
                raise
            with _lock:
                _cache[key] = (time.time(), val)
            STATUS[source] = {"ok": True, "checked_at": _now_iso(), "error": None}
            return val
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


def _get(url: str, params: Optional[dict] = None, timeout: int = 45, **kw) -> requests.Response:
    r = requests.get(url, params=params, headers={**HEADERS, **kw.pop("headers", {})}, timeout=timeout, **kw)
    r.raise_for_status()
    return r


def _arcgis_geojson(layer_url: str, bbox: Bbox, out_fields: str, where: str = "1=1",
                    offset_deg: float = 0.0003, max_records: int = 2000) -> dict:
    params = {"where": where, "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
              "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": out_fields, "outSR": 4326,
              "returnGeometry": "true", "maxAllowableOffset": offset_deg, "resultRecordCount": max_records,
              "f": "geojson"}
    d = _get(f"{layer_url}/query", params).json()
    if "error" in d:
        raise RuntimeError(d["error"])
    return d


def _ms_to_iso(ms) -> Optional[str]:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(timespec="minutes") if ms else None


# ----------------------------------------------------------------------------- fire perimeters and incidents
WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0"
CAL_HIST = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Historic_Fire_Perimeters/FeatureServer/2"
CAL_INC = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList"


@cached("nifc_wfigs_current", ttl=600)
def current_perimeters(bbox: Bbox) -> dict:
    d = _arcgis_geojson(WFIGS, bbox, "poly_IncidentName,poly_GISAcres,attr_PercentContained,"
                        "attr_FireDiscoveryDateTime,poly_DateCurrent,attr_IncidentTypeCategory,attr_POOState")
    for f in d.get("features", []):
        p = f["properties"]
        f["properties"] = {"name": p.get("poly_IncidentName"), "acres": p.get("poly_GISAcres"),
                           "percent_contained": p.get("attr_PercentContained"),
                           "discovered": _ms_to_iso(p.get("attr_FireDiscoveryDateTime")),
                           "perimeter_updated": _ms_to_iso(p.get("poly_DateCurrent")),
                           "type": p.get("attr_IncidentTypeCategory"), "kind": "current",
                           "source": "NIFC WFIGS Interagency Perimeters (current)"}
    return d


@cached("calfire_historic_perimeters", ttl=86400)
def historic_perimeters(bbox: Bbox, since_year: int = 1950) -> dict:
    d = _arcgis_geojson(CAL_HIST, bbox, "FIRE_NAME,YEAR_,GIS_ACRES,ALARM_DATE,CONT_DATE,CAUSE",
                        where=f"YEAR_ >= {int(since_year)}", offset_deg=0.0005)
    for f in d.get("features", []):
        p = f["properties"]
        f["properties"] = {"name": (p.get("FIRE_NAME") or "").title(), "year": p.get("YEAR_"),
                           "acres": p.get("GIS_ACRES"), "alarm_date": _ms_to_iso(p.get("ALARM_DATE")),
                           "kind": "historic", "source": "CAL FIRE FRAP California Fire Perimeters (1950+)"}
    return d


@cached("calfire_incidents", ttl=600)
def calfire_incidents(bbox: Bbox = CA_BBOX) -> dict:
    d = _get(CAL_INC, {"inactive": "false"}, headers={"User-Agent": "Mozilla/5.0"}).json()
    feats = []
    for f in d.get("features", []):
        lon, lat = f["geometry"]["coordinates"][:2]
        if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        p = f["properties"]
        feats.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "name": (p.get("Name") or "").strip(), "acres": p.get("AcresBurned"),
            "percent_contained": p.get("PercentContained"), "county": p.get("County"),
            "location": p.get("Location"), "started": p.get("Started"), "updated": p.get("Updated"),
            "admin_unit": p.get("AdminUnit"), "url": p.get("Url"), "type": p.get("Type"),
            "kind": "incident", "source": "CAL FIRE incidents (incidents.fire.ca.gov)"}})
    return {"type": "FeatureCollection", "features": feats}


# ----------------------------------------------------------------------------- NASA FIRMS hotspots
FIRMS_FILES = {
    "VIIRS S-NPP": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
    "VIIRS NOAA-20": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
    "VIIRS NOAA-21": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
}


@cached("nasa_firms_viirs", ttl=900)
def firms_hotspots(bbox: Bbox = CA_BBOX) -> dict:
    feats, errors = [], []
    for sensor, url in FIRMS_FILES.items():
        try:
            text = _get(url, timeout=60).text
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sensor}: {e}")
            continue
        for r in csv.DictReader(io.StringIO(text)):
            lat, lon = float(r["latitude"]), float(r["longitude"])
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
            t = r["acq_time"].zfill(4)
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {
                "sensor": sensor, "brightness_ti4_k": float(r["bright_ti4"]), "frp_mw": float(r["frp"]),
                "confidence": r["confidence"], "day_night": r["daynight"],
                "acquired_utc": f"{r['acq_date']}T{t[:2]}:{t[2:]}Z", "source": "NASA FIRMS (VIIRS 375 m, last 24 h)"}})
    if errors and not feats and len(errors) == len(FIRMS_FILES):
        raise RuntimeError("; ".join(errors))
    return {"type": "FeatureCollection", "features": feats, "errors": errors}


# ----------------------------------------------------------------------------- CDEC RAWS fuel moisture
CDEC = "https://cdec.water.ca.gov"
CDEC_SENSORS = {11: "fuel_moisture_10h_pct", 4: "air_temp_f", 12: "rel_humidity_pct", 9: "wind_speed_mph",
                10: "wind_dir_deg", 77: "wind_gust_mph"}


@cached("cdec_station_list", ttl=86400)
def cdec_fuel_stations(bbox: Bbox) -> List[dict]:
    params = {"sta": "", "sensor_chk": "on", "sensor": 11, "collect": "NONE SPECIFIED", "dur": "", "active_chk": "on",
              "active": "Y", "loc_chk": "on", "lon1": bbox[0], "lon2": bbox[2], "lat1": bbox[1], "lat2": bbox[3],
              "elev1": -5, "elev2": 99000, "nearby": "", "basin": "NONE SPECIFIED", "hydro": "NONE SPECIFIED",
              "county": "NONE SPECIFIED", "agency_num": 160, "display": "sta"}
    t = _get(f"{CDEC}/dynamicapp/staSearch", params, timeout=60).text
    out = []
    for row in re.findall(r"<tr>(.*?)</tr>", t, re.S):
        c = [html.unescape(re.sub("<[^>]+>", "", x)).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(c) >= 8 and re.fullmatch(r"[A-Z0-9]{3}", c[0]):
            out.append({"id": c[0], "name": c[1].title(), "county": c[3].title(), "lon": float(c[4]), "lat": float(c[5]),
                        "elevation_ft": c[6], "operator": c[7]})
    return out


@cached("cdec_raws_fuel_moisture", ttl=900)
def fuel_moisture(bbox: Bbox) -> dict:
    stations = cdec_fuel_stations(bbox)
    if not stations:
        return {"type": "FeatureCollection", "features": []}
    end = dt.date.today() + dt.timedelta(days=1)
    rows = _get(f"{CDEC}/dynamicapp/req/JSONDataServlet", {
        "Stations": ",".join(s["id"] for s in stations), "SensorNums": ",".join(map(str, CDEC_SENSORS)),
        "dur_code": "H", "Start": str(end - dt.timedelta(days=3)), "End": str(end)}, timeout=90).json()
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        v = r.get("value")
        if v is None or v == -9999:
            continue
        key = CDEC_SENSORS.get(r.get("SENSOR_NUM") or r.get("sensorNumber"))
        if not key:
            continue
        cur = latest.setdefault(r["stationId"], {})
        t = dt.datetime.strptime(r["date"], "%Y-%m-%d %H:%M")  # hours are not zero-padded; compare as datetimes
        if key not in cur or t >= cur[key][0]:
            cur[key] = (t, v)
    feats = []
    for s in stations:
        obs = latest.get(s["id"], {})
        props = {"station_id": s["id"], "station_name": s["name"], "county": s["county"], "operator": s["operator"],
                 "elevation_ft": s["elevation_ft"], "source": "CDEC RAWS (sensor 11: 10-h fuel stick moisture)"}
        for k, (t, v) in obs.items():
            props[k] = v
            props[k + "_time_pst"] = t.strftime("%Y-%m-%d %H:%M")
        fm = props.get("fuel_moisture_10h_pct")
        props["dryness_class"] = (None if fm is None else "critically dry (<=6%)" if fm <= 6 else
                                  "dry (7-10%)" if fm <= 10 else "moderate (11-15%)" if fm <= 15 else "moist (>15%)")
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats,
            "note": "CDEC reports times in Pacific Standard Time. Thresholds are a common rule of thumb for 10-h fuels."}


# ----------------------------------------------------------------------------- weather (HRRR) and alerts
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def cardinal(deg: Optional[float]) -> Optional[str]:
    return None if deg is None else COMPASS[int(((deg % 360) + 11.25) // 22.5) % 16]


@cached("noaa_hrrr_open_meteo", ttl=600)
def current_weather(lat: float, lon: float) -> dict:
    params = {"latitude": lat, "longitude": lon, "models": "gfs_hrrr", "wind_speed_unit": "mph",
              "temperature_unit": "fahrenheit", "timezone": "UTC",
              "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m"}
    d = _get(OPEN_METEO, params).json()
    c = d["current"]
    return {"source": "NOAA HRRR (3 km) via Open-Meteo", "valid_time_utc": c["time"] + "Z",
            "grid_lat": d["latitude"], "grid_lon": d["longitude"],
            "temperature_f": c["temperature_2m"], "relative_humidity_pct": c["relative_humidity_2m"],
            "wind_speed_mph": c["wind_speed_10m"], "wind_gust_mph": c["wind_gusts_10m"],
            "wind_direction_deg": c["wind_direction_10m"], "wind_direction_cardinal": cardinal(c["wind_direction_10m"])}


@cached("noaa_hrrr_wind_grid", ttl=900)
def wind_grid(bbox: Bbox, n: int = 6) -> dict:
    lats = np.linspace(bbox[1], bbox[3], n + 2)[1:-1]
    lons = np.linspace(bbox[0], bbox[2], n + 2)[1:-1]
    pts = [(round(float(a), 4), round(float(o), 4)) for a in lats for o in lons]
    params = {"latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
              "models": "gfs_hrrr", "wind_speed_unit": "mph", "timezone": "UTC",
              "current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m"}
    d = _get(OPEN_METEO, params).json()
    d = d if isinstance(d, list) else [d]
    feats = []
    for (lat, lon), r in zip(pts, d):
        c = r["current"]
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {
            "wind_speed_mph": c["wind_speed_10m"], "wind_gust_mph": c["wind_gusts_10m"],
            "wind_from_deg": c["wind_direction_10m"], "valid_time_utc": c["time"] + "Z"}})
    return {"type": "FeatureCollection", "features": feats, "source": "NOAA HRRR (3 km) via Open-Meteo, 10 m wind"}


@cached("nws_alerts", ttl=600)
def nws_alerts(lat: float, lon: float) -> List[dict]:
    d = _get("https://api.weather.gov/alerts/active", {"point": f"{lat:.4f},{lon:.4f}"},
             headers={"Accept": "application/geo+json"}).json()
    return [{"event": f["properties"].get("event"), "headline": f["properties"].get("headline"),
             "severity": f["properties"].get("severity"), "onset": f["properties"].get("onset"),
             "ends": f["properties"].get("ends") or f["properties"].get("expires")}
            for f in d.get("features", [])]


# ----------------------------------------------------------------------------- terrain slope (USGS 3DEP)
DEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"


def _bbox_key(bbox: Bbox) -> str:
    return hashlib.md5(",".join(f"{v:.4f}" for v in bbox).encode()).hexdigest()[:10]


@cached("usgs_3dep", ttl=86400)
def slope_overlay(bbox: Bbox, size: int = 700) -> dict:
    import rasterio
    from PIL import Image
    w = size
    h = max(64, int(size * (bbox[3] - bbox[1]) / ((bbox[2] - bbox[0]) * math.cos(math.radians((bbox[1] + bbox[3]) / 2)))))
    r = _get(DEP, {"bbox": ",".join(map(str, bbox)), "bboxSR": 4326, "imageSR": 4326, "size": f"{w},{h}",
                   "format": "tiff", "pixelType": "F32", "f": "image"}, timeout=90)
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        z = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        z[z == nodata] = np.nan
    lat0 = (bbox[1] + bbox[3]) / 2
    dx = (bbox[2] - bbox[0]) / z.shape[1] * 111320.0 * math.cos(math.radians(lat0))
    dy = (bbox[3] - bbox[1]) / z.shape[0] * 110574.0
    gy, gx = np.gradient(z, dy, dx)
    slope_pct = np.hypot(gx, gy) * 100.0
    rgba = np.zeros(z.shape + (4,), dtype=np.uint8)
    rgba[(slope_pct >= 15) & (slope_pct < 30)] = [250, 204, 21, 110]
    rgba[(slope_pct >= 30) & (slope_pct < 50)] = [249, 115, 22, 140]
    rgba[slope_pct >= 50] = [220, 38, 38, 170]
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    name = f"slope_{_bbox_key(bbox)}.png"
    Image.fromarray(rgba, mode="RGBA").save(STATIC_DIR / name)
    valid = ~np.isnan(slope_pct)
    share = {k: round(float(np.mean(m[valid])) * 100, 1) for k, m in
             {"lt15": slope_pct < 15, "15_30": (slope_pct >= 15) & (slope_pct < 30),
              "30_50": (slope_pct >= 30) & (slope_pct < 50), "ge50": slope_pct >= 50}.items()}
    return {"image_url": f"/static/{name}", "bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            "pixel_m": round((dx + dy) / 2, 1), "area_share_pct": share,
            "source": "USGS 3DEP elevation; slope (%) computed from the DEM",
            "classes": {"15-30%": "#FACC15", "30-50%": "#F97316", ">=50%": "#DC2626"}}


def slope_at(lat: float, lon: float) -> Optional[float]:
    d = 0.002
    try:
        ov = slope_overlay((lon - d, lat - d, lon + d, lat + d), size=64)
    except Exception:  # noqa: BLE001
        return None
    s = ov["area_share_pct"]
    return s  # share of slope classes within ~200 m


# ----------------------------------------------------------------------------- Sentinel-2 NDVI
S2 = "https://sentinel.arcgis.com/arcgis/rest/services/Sentinel2/ImageServer"


@cached("sentinel2_ndvi", ttl=21600)
def ndvi_overlay(bbox: Bbox, max_cloud: float = 0.1, size: int = 900) -> dict:
    from PIL import Image
    q = _get(f"{S2}/query", {"where": f"cloudcover <= {max_cloud}", "geometry": ",".join(map(str, bbox)),
                             "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                             "spatialRel": "esriSpatialRelIntersects", "outFields": "objectid,acquisitiondate,cloudcover,name",
                             "orderByFields": "acquisitiondate DESC", "resultRecordCount": 1, "returnGeometry": "false",
                             "f": "json"}).json()
    feats = q.get("features") or []
    if not feats:
        raise RuntimeError("no Sentinel-2 scene with cloud cover <= %.0f%% in the last 14 months" % (max_cloud * 100))
    a = feats[0]["attributes"]
    h = max(64, int(size * (bbox[3] - bbox[1]) / ((bbox[2] - bbox[0]) * math.cos(math.radians((bbox[1] + bbox[3]) / 2)))))
    r = _get(f"{S2}/exportImage", {
        "bbox": ",".join(map(str, bbox)), "bboxSR": 4326, "imageSR": 4326, "size": f"{size},{h}", "format": "png",
        "renderingRule": json.dumps({"rasterFunction": "NDVI Colormap"}),
        "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [a["objectid"]]}),
        "f": "image"}, timeout=90)
    im = Image.open(io.BytesIO(r.content)).convert("RGBA")
    arr = np.array(im)
    arr[(arr[..., :3] == 0).all(-1)] = 0  # transparent outside the scene footprint
    name = f"ndvi_{_bbox_key(bbox)}_{a['objectid']}.png"
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(STATIC_DIR / name)
    return {"image_url": f"/static/{name}", "bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            "scene": a["name"], "acquired_utc": _ms_to_iso(a["acquisitiondate"]),
            "cloud_cover_pct": round(a["cloudcover"] * 100, 1),
            "source": "Copernicus Sentinel-2 L2A via Esri Sentinel-2 Views (NDVI colormap)"}


# ----------------------------------------------------------------------------- CAL FIRE FHSZ
FHSZ_LAYERS = {
    "SRA (2023)": "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/FHSZSRA_23_3/FeatureServer/0",
    "LRA (2025)": "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/FHSALRA25_v1_All/FeatureServer/0",
}
FHSZ_COLORS = {"Moderate": "#EAB308", "High": "#F97316", "Very High": "#DC2626"}


@cached("calfire_fhsz", ttl=86400)
def fhsz(bbox: Bbox) -> dict:
    feats = []
    for label, url in FHSZ_LAYERS.items():
        d = _arcgis_geojson(url, bbox, "SRA,FHSZ,FHSZ_Description", where="FHSZ >= 1", offset_deg=0.0002)
        for f in d.get("features", []):
            p = f["properties"]
            f["properties"] = {"zone": p.get("FHSZ_Description"), "responsibility_area": label,
                               "color": FHSZ_COLORS.get(p.get("FHSZ_Description"), "#a3a3a3"),
                               "source": f"CAL FIRE Fire Hazard Severity Zones, {label}"}
            feats.append(f)
    return {"type": "FeatureCollection", "features": feats}


def fhsz_at(lat: float, lon: float) -> Optional[dict]:
    for label, url in FHSZ_LAYERS.items():
        d = _get(f"{url}/query", {"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
                                  "spatialRel": "esriSpatialRelIntersects", "outFields": "FHSZ,FHSZ_Description",
                                  "returnGeometry": "false", "f": "json"}).json()
        for f in d.get("features", []):
            if (f["attributes"].get("FHSZ") or 0) >= 1:
                return {"zone": f["attributes"]["FHSZ_Description"], "responsibility_area": label}
    return {"zone": "Not in a mapped FHSZ (Moderate or higher)", "responsibility_area": None}


# ----------------------------------------------------------------------------- USDA Wildfire Risk to Communities
# Annual burn probability (FSim, 270 m upsampled to 30 m; landscape as of end-2020), 2024 release
# (Scott et al. 2024, https://doi.org/10.2737/RDS-2020-0016-2). The public image service stores BP x 10,000 as
# U16: the published BP range is 0-0.14 and the service maximum is 1352. The service is meant for display; the
# source rasters are the reference for quantitative work.
WRC_BP = ("https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/"
          "USFS_EDW_RMRS_WRC_BurnProbability/ImageServer")
WRC_SCALE = 10000.0
WRC_CLASSES = [  # (lower bound of annual BP, label, RGBA); below 0.0005 is left transparent
    (0.0005, "0.05-0.2 %/yr", (191, 219, 254, 130)),
    (0.002, "0.2-0.5 %/yr", (96, 165, 250, 160)),
    (0.005, "0.5-1 %/yr", (37, 99, 235, 185)),
    (0.01, ">=1 %/yr", (30, 58, 138, 210)),
]
WRC_SOURCE = "USDA Forest Service, Wildfire Risk to Communities (2024 release), annual burn probability"


@cached("usda_wrc_bp", ttl=86400)
def wrc_bp_overlay(bbox: Bbox, size: int = 900) -> dict:
    import rasterio
    from PIL import Image
    w = size
    h = max(64, int(size * (bbox[3] - bbox[1]) / ((bbox[2] - bbox[0]) * math.cos(math.radians((bbox[1] + bbox[3]) / 2)))))
    r = _get(f"{WRC_BP}/exportImage", {"bbox": ",".join(map(str, bbox)), "bboxSR": 4326, "imageSR": 4326,
                                       "size": f"{w},{h}", "format": "tiff", "pixelType": "U16",
                                       "interpolation": "RSP_NearestNeighbor", "f": "image"}, timeout=90)
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        v = src.read(1).astype(float)
        b = src.bounds  # the service may widen the bbox to keep the aspect ratio
    bp = v / WRC_SCALE
    rgba = np.zeros(bp.shape + (4,), dtype=np.uint8)
    for lo, _, col in WRC_CLASSES:
        rgba[bp >= lo] = col
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    name = f"wrc_bp_{_bbox_key(bbox)}.png"
    Image.fromarray(rgba, mode="RGBA").save(STATIC_DIR / name)
    burnable = bp > 0
    bounds = [lo for lo, _, _ in WRC_CLASSES] + [1.0]
    share = {"lt0.05": round(float(np.mean(bp[burnable] < WRC_CLASSES[0][0])) * 100, 1) if burnable.any() else None}
    for (lo, label, _), hi in zip(WRC_CLASSES, bounds[1:]):
        share[label] = round(float(np.mean((bp[burnable] >= lo) & (bp[burnable] < hi))) * 100, 1) if burnable.any() else None
    return {"image_url": f"/static/{name}", "bounds": [[b.bottom, b.left], [b.top, b.right]],
            "max_annual_bp": round(float(bp.max()), 4), "median_annual_bp_nonzero":
                round(float(np.median(bp[burnable])), 4) if burnable.any() else None,
            "area_share_pct_of_nonzero": share, "pixel_m": 30, "landscape_as_of": "end of 2020 (LANDFIRE 2020)",
            "source": WRC_SOURCE, "citation": "Scott et al. 2024, doi:10.2737/RDS-2020-0016-2",
            "classes": {label: "#%02X%02X%02X" % col[:3] for _, label, col in WRC_CLASSES}}


def wrc_bp_at(lat: float, lon: float) -> Optional[dict]:
    d = _get(f"{WRC_BP}/identify", {"geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
                                    "geometryType": "esriGeometryPoint", "returnGeometry": "false",
                                    "returnCatalogItems": "false", "f": "json"}).json()
    val = d.get("value")
    if val in (None, "", "NoData"):
        return None
    bp = float(val) / WRC_SCALE
    return {"annual_burn_probability": round(bp, 5), "pct_per_year": round(bp * 100, 3),
            "one_in_n_years": round(1 / bp) if bp > 0 else None, "source": WRC_SOURCE,
            "landscape_as_of": "end of 2020"}


def historic_fires_at(lat: float, lon: float) -> List[dict]:
    d = _get(f"{CAL_HIST}/query", {"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
                                   "spatialRel": "esriSpatialRelIntersects", "outFields": "FIRE_NAME,YEAR_,GIS_ACRES",
                                   "orderByFields": "YEAR_ DESC", "returnGeometry": "false", "f": "json"}).json()
    return [{"name": (f["attributes"]["FIRE_NAME"] or "").title(), "year": f["attributes"]["YEAR_"],
             "acres": f["attributes"]["GIS_ACRES"]} for f in d.get("features", [])]


# ----------------------------------------------------------------------------- ESA WorldCover land cover
WORLDCOVER = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
WC_CLASSES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
              60: "Bare / sparse vegetation", 70: "Snow and ice", 80: "Permanent water", 90: "Herbaceous wetland",
              95: "Mangroves", 100: "Moss and lichen"}


@cached("esa_worldcover", ttl=86400)
def land_cover_around(lat: float, lon: float, radius_m: float = 100.0) -> dict:
    import rasterio
    from rasterio.windows import from_bounds
    la, lo = int(math.floor(lat / 3) * 3), int(math.floor(lon / 3) * 3)
    tile = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    dlat, dlon = radius_m / 110574.0, radius_m / (111320.0 * math.cos(math.radians(lat)))
    with rasterio.open(WORLDCOVER.format(tile=tile)) as src:
        a = src.read(1, window=from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform))
        centre = int(a[a.shape[0] // 2, a.shape[1] // 2]) if a.size else None
    vals, cnt = np.unique(a, return_counts=True)
    share = {WC_CLASSES.get(int(v), str(v)): round(float(c) / a.size * 100, 1) for v, c in zip(vals, cnt)}
    return {"at_point": WC_CLASSES.get(centre), "within_radius_pct": dict(sorted(share.items(), key=lambda x: -x[1])),
            "radius_m": radius_m, "source": "ESA WorldCover 2021 (10 m)"}


# ----------------------------------------------------------------------------- Microsoft building footprints
_buildings = None


def _load_buildings():
    global _buildings
    if _buildings is None:
        import pandas as pd
        if not BUILDINGS_PARQUET.exists():
            raise FileNotFoundError(f"{BUILDINGS_PARQUET} missing; run scripts/data/extract_ms_buildings_region.py")
        _buildings = pd.read_parquet(BUILDINGS_PARQUET)
    return _buildings


def buildings(bbox: Bbox, limit: int = 4000) -> dict:
    try:
        df = _load_buildings()
    except Exception as e:  # noqa: BLE001
        STATUS["ms_building_footprints"] = {"ok": False, "checked_at": _now_iso(), "error": str(e)}
        raise
    m = (df["lon"] >= bbox[0]) & (df["lon"] <= bbox[2]) & (df["lat"] >= bbox[1]) & (df["lat"] <= bbox[3])
    sub = df[m]
    n_total = int(len(sub))
    if n_total > limit:
        sub = sub.sample(n=limit, random_state=0)
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [json.loads(r)]},
              "properties": {"area_m2": a, "area_sqft": round(a * 10.7639), "source": "Microsoft US Building Footprints"}}
             for r, a in zip(sub["ring"], sub["area_m2"])]
    STATUS["ms_building_footprints"] = {"ok": True, "checked_at": _now_iso(), "error": None}
    return {"type": "FeatureCollection", "features": feats, "n_in_view": n_total, "truncated": n_total > limit}


def buildings_within(lat: float, lon: float, radius_m: float) -> dict:
    df = _load_buildings()
    kx, ky = 111320.0 * math.cos(math.radians(lat)), 110574.0
    d = np.hypot((df["lon"].to_numpy() - lon) * kx, (df["lat"].to_numpy() - lat) * ky)
    m = d <= radius_m
    return {"count": int(m.sum()), "footprint_area_m2": float(df.loc[m, "area_m2"].sum()),
            "nearest_m": float(d.min()) if len(d) else None}


# ----------------------------------------------------------------------------- source registry for the UI
SOURCES = [
    {"id": "bellwether", "name": "Bellwether wildfire hazard model (1-year and 5-year)", "provider": "Bellwether, X (Alphabet)",
     "access": "GeoTIFFs shared with the pilot partners (local files); covers the San Mateo / Santa Cruz pilot area only",
     "resolution": "100 m", "cadence": "Quarterly releases",
     "description": "Absolute probability that a 100 m cell lies inside a wildfire perimeter within the next 1 or 5 years. "
                    "Per the release docs, it does not model real-time spread or structure vulnerability."},
    {"id": "nifc_wfigs_current", "name": "Current wildfire perimeters", "provider": "NIFC WFIGS (interagency)",
     "access": "Public ArcGIS feature service", "resolution": "Vector", "cadence": "As incidents update",
     "description": "Perimeters of active and recent incidents reported to IRWIN."},
    {"id": "calfire_historic_perimeters", "name": "Historical fire perimeters (1950+)", "provider": "CAL FIRE FRAP",
     "access": "Public ArcGIS feature service", "resolution": "Vector", "cadence": "Annual",
     "description": "Final perimeters of past fires in California."},
    {"id": "calfire_incidents", "name": "Active CAL FIRE incidents", "provider": "CAL FIRE",
     "access": "incidents.fire.ca.gov GeoJSON", "resolution": "Points", "cadence": "As incidents update",
     "description": "Incident locations with acres burned and containment."},
    {"id": "nasa_firms_viirs", "name": "Satellite thermal hotspots (last 24 h)", "provider": "NASA FIRMS",
     "access": "Public 24 h CSV files (no key)", "resolution": "375 m (VIIRS)", "cadence": "Several overpasses per day",
     "description": "Active-fire detections from VIIRS on S-NPP, NOAA-20 and NOAA-21."},
    {"id": "cdec_raws_fuel_moisture", "name": "RAWS fuel moisture and weather stations", "provider": "California Data Exchange Center (CDEC)",
     "access": "CDEC JSON data servlet", "resolution": "Stations", "cadence": "Hourly",
     "description": "10-h fuel stick moisture (sensor 11), air temperature, humidity and wind at RAWS stations."},
    {"id": "noaa_hrrr_open_meteo", "name": "Current weather and wind", "provider": "NOAA HRRR via Open-Meteo",
     "access": "Open-Meteo forecast API (models=gfs_hrrr)", "resolution": "3 km", "cadence": "Hourly",
     "description": "Temperature, humidity, 10 m wind and gusts; the wind layer samples a grid over the view."},
    {"id": "nws_alerts", "name": "Weather alerts (Red Flag Warnings)", "provider": "National Weather Service",
     "access": "api.weather.gov", "resolution": "Zones", "cadence": "As issued",
     "description": "Active watches and warnings at the selected location."},
    {"id": "usgs_3dep", "name": "Terrain slope", "provider": "USGS 3D Elevation Program",
     "access": "3DEPElevation ImageServer (DEM); slope computed in this app", "resolution": "~10-30 m as requested",
     "cadence": "Static", "description": "Slope in percent from the 3DEP DEM, classed at 15 / 30 / 50 %."},
    {"id": "sentinel2_ndvi", "name": "Vegetation greenness (NDVI)", "provider": "Copernicus Sentinel-2 via Esri Sentinel-2 Views",
     "access": "Public ArcGIS image service", "resolution": "10 m", "cadence": "~5 days (latest scene with <10% cloud)",
     "description": "NDVI from one dated Sentinel-2 scene; the acquisition date is shown on the map."},
    {"id": "calfire_fhsz", "name": "Fire Hazard Severity Zones", "provider": "CAL FIRE",
     "access": "Public ArcGIS feature services (SRA 2023, LRA 2025)", "resolution": "Vector", "cadence": "Map revisions",
     "description": "Official Moderate / High / Very High zones in State and Local Responsibility Areas."},
    {"id": "usda_wrc_bp", "name": "Annual burn probability (Wildfire Risk to Communities)", "provider": "USDA Forest Service",
     "access": "Public ArcGIS image service (geoplatform.gov); data: doi:10.2737/RDS-2020-0016-2", "resolution": "30 m (FSim at 270 m, upsampled)",
     "cadence": "2024 release; landscape as of end-2020",
     "description": "Simulated yearly chance that a fire burns each location (FSim). Public, national counterpart to the Bellwether layer."},
    {"id": "esa_worldcover", "name": "Land cover", "provider": "ESA WorldCover 2021",
     "access": "Cloud-optimised GeoTIFFs on AWS", "resolution": "10 m", "cadence": "2021 map",
     "description": "Used for the land cover around a clicked point."},
    {"id": "ms_building_footprints", "name": "Building footprints", "provider": "Microsoft US Building Footprints",
     "access": "Downloaded California release, extracted for the Bay Area", "resolution": "Polygons",
     "cadence": "Static release (imagery mostly 2018-2020)", "description": "Footprint polygons and area; no address or assessor data."},
]


def check_all(region_bbox: Bbox, lat: float, lon: float) -> List[dict]:
    """Fetch every source once (in parallel) and report what worked."""
    from concurrent.futures import ThreadPoolExecutor
    probes = {
        "nifc_wfigs_current": lambda: len(current_perimeters(CA_BBOX)["features"]),
        "calfire_historic_perimeters": lambda: len(historic_perimeters(region_bbox)["features"]),
        "calfire_incidents": lambda: len(calfire_incidents()["features"]),
        "nasa_firms_viirs": lambda: len(firms_hotspots()["features"]),
        "cdec_raws_fuel_moisture": lambda: len(fuel_moisture((region_bbox[0] - 0.3, region_bbox[1] - 0.3,
                                                              region_bbox[2] + 0.3, region_bbox[3] + 0.3))["features"]),
        "noaa_hrrr_open_meteo": lambda: current_weather(lat, lon)["valid_time_utc"],
        "nws_alerts": lambda: len(nws_alerts(lat, lon)),
        "usgs_3dep": lambda: slope_overlay(region_bbox)["pixel_m"],
        "sentinel2_ndvi": lambda: ndvi_overlay(region_bbox)["acquired_utc"],
        "calfire_fhsz": lambda: len(fhsz(region_bbox)["features"]),
        "usda_wrc_bp": lambda: wrc_bp_overlay(region_bbox)["max_annual_bp"],
        "esa_worldcover": lambda: land_cover_around(lat, lon)["at_point"],
        "ms_building_footprints": lambda: buildings_within(lat, lon, 500)["count"],
    }

    def run(item):
        sid, fn = item
        t0 = time.time()
        try:
            return sid, {"ok": True, "sample": fn(), "latency_s": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            return sid, {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "latency_s": round(time.time() - t0, 2)}

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = dict(ex.map(run, probes.items()))
    out = []
    for s in SOURCES:
        r = res.get(s["id"], {"ok": None})
        out.append({**s, "status": r, "checked_at": _now_iso()})
    return out
