"""
Statewide vector layers: ingest (from ArcGIS feature services and the local Microsoft footprints), spatial queries,
and drawing into map tiles. Stored as GeoParquet in EPSG:4326 under VECTORS. Building footprints (about 11 million)
are split into 0.25-degree partitions so a query reads only the files it needs; the other layers are loaded once
into memory with a spatial index.
"""
import json
import logging
import math
import threading
import time
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box

from backend.geodata.catalog import VECTOR_LAYERS, layer
from backend.geodata.config import RAW, UA, VECTORS

log = logging.getLogger("geodata")
CA_BBOX = (-124.6, 32.4, -114.0, 42.1)
PART = 0.25  # degrees, building partitions


# ----------------------------------------------------------------------------- ingest
def arcgis_features(url, where="1=1", fields="*", page=1000, sleep=0.2):
    """All features of a feature-service layer as a GeoDataFrame (EPSG:4326), with coded values decoded."""
    meta = requests.get(url, params={"f": "json"}, headers=UA, timeout=60).json()
    domains = {f["name"]: {c["code"]: c["name"] for c in (f.get("domain") or {}).get("codedValues", [])}
               for f in meta.get("fields", []) if (f.get("domain") or {}).get("codedValues")}
    page = min(page, meta.get("maxRecordCount", page))
    ids = requests.get(f"{url}/query", params={"where": where, "returnIdsOnly": "true", "f": "json"}, headers=UA,
                       timeout=120).json()
    oid_field = ids.get("objectIdFieldName", "OBJECTID")
    oids = sorted(ids.get("objectIds") or [])
    frames = []
    for k in range(0, len(oids), page):
        chunk = oids[k:k + page]
        for attempt in range(4):
            try:
                r = requests.post(f"{url}/query", headers=UA, timeout=300, data={
                    "where": f"{oid_field} >= {chunk[0]} AND {oid_field} <= {chunk[-1]} AND ({where})",
                    "outFields": fields, "outSR": 4326, "f": "geojson"})
                fc = r.json()
                if "features" not in fc:
                    raise RuntimeError(str(fc)[:200])
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        if fc["features"]:
            frames.append(gpd.GeoDataFrame.from_features(fc["features"], crs=4326))
        time.sleep(sleep)
    g = pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame(geometry=[], crs=4326)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    for col, dom in domains.items():
        if col in g:
            g[col + "_label"] = g[col].map(dom)
    return g


def _clean(g):
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.make_valid()
    for c in g.columns:  # epoch-millisecond dates -> ISO
        datey = any(t in c.upper() for t in ("DATE", "_START", "_END", "TIME")) or c.endswith("_DT")
        if c != "geometry" and g[c].dtype.kind in "if" and datey:
            if g[c].dropna().between(-2.2e12, 4.1e12).all() and g[c].dropna().abs().max() > 1e9:
                g[c] = pd.to_datetime(g[c], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    return g


def _save(lid, g):
    VECTORS.mkdir(parents=True, exist_ok=True)
    p = VECTORS / VECTOR_LAYERS[lid]["file"]
    g = g.to_crs(4326)
    # spatial ordering + small row groups so bbox reads prune well
    c = g.geometry.representative_point()
    g = g.iloc[np.lexsort((np.floor(c.x / 0.1), np.floor(c.y / 0.1)))]
    g.to_parquet(p, write_covering_bbox=True, row_group_size=20000)
    (p.with_suffix(".meta.json")).write_text(json.dumps({"features": len(g), "built_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    load.cache_clear()
    return len(g)


CF = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services"
NIFC = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"


def ingest_fhsz():
    parts = []
    for url, resp in [(f"{CF}/FHSZSRA_23_3/FeatureServer/0", "SRA (state)"),
                      (f"{CF}/FHSALRA25_v1_All/FeatureServer/0", "LRA (local)")]:
        g = arcgis_features(url, fields="FHSZ,FHSZ_Description")
        g["responsibility_area"] = resp
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    g = g[g.FHSZ_Description.isin(["Moderate", "High", "Very High"])]
    return _save("fhsz", _clean(gpd.GeoDataFrame(g, crs=4326)))


def ingest_fire_history():
    g = arcgis_features(f"{CF}/California_Historic_Fire_Perimeters/FeatureServer/2",
                        fields="YEAR_,FIRE_NAME,ALARM_DATE,CONT_DATE,CAUSE,GIS_ACRES,AGENCY,UNIT_ID,INC_NUM")
    return _save("fire_history", _clean(g))


def ingest_ignitions():
    keep = ("IncidentName,FireDiscoveryDateTime,CalendarYear,FinalAcres,IncidentSize,FireCauseID,FireCauseGeneralID,"
            "POOProtectingAgency,POOLandownerCategory,POOCounty,PredominantFuelGroup,UniqueFireIdentifier,"
            "IncidentTypeCategory")
    g = arcgis_features(f"{NIFC}/InFORM_FireOccurrence_Public/FeatureServer/0",
                        where="POOState='US-CA' AND IncidentTypeCategory='WF'", fields=keep)
    g["cause"] = g.get("FireCauseID_label")
    g["cause_detail"] = g.get("FireCauseGeneralID_label")
    g["acres"] = g["FinalAcres"].fillna(g["IncidentSize"])
    g["discovered"] = pd.to_datetime(g["FireDiscoveryDateTime"], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ("POOLandownerCategory", "PredominantFuelGroup"):  # keep the readable label only
        if c + "_label" in g:
            g[c] = g.pop(c + "_label")
    g = g.drop(columns=[c for c in ("FireCauseID", "FireCauseGeneralID", "FireCauseID_label", "FireCauseGeneralID_label",
                                    "FinalAcres", "IncidentSize", "IncidentTypeCategory", "IncidentTypeCategory_label",
                                    "FireDiscoveryDateTime") if c in g])
    return _save("ignitions", _clean(g))


def ingest_treatments():
    n = {}
    g = arcgis_features(f"{CF}/FTER%20Treatment%20Polygons%20Public%20View/FeatureServer/0",
                        fields="REGIONUNIT,COUNTY_NAME,PROJECT_TYPE,CALMAPPER_ID,PROJECT_NAME,TREATMENT_NAME,"
                               "TREATMENT_OBJECTIVE,PROJECT_STATUS,ACTIVITY_STATUS,PROJECT_START,PROJECT_END,"
                               "ACTIVITY_START,ACTIVITY_END,ALL_ACTIVITIES,BROAD_VEG_TYPES,"
                               "TREATMENT_FOOTPRINT_ACRES,TREATED_ACRES_LIFETIME,FUNDING_SOURCES")
    n["treatments_calfire"] = _save("treatments_calfire", _clean(g))
    g = arcgis_features(f"{CF}/California_Prescribed_Fire_Perimeters_Public/FeatureServer/0",
                        fields="YEAR_,AGENCY,UNIT_ID,TREATMENT_ID,TREATMENT_NAME,TREATMENT_TYPE,START_DATE,END_DATE,"
                               "TREATED_AC,GIS_ACRES")
    n["treatments_rx"] = _save("treatments_rx", _clean(g))
    # CalVTP: contact name / phone / e-mail / address and editor fields are personal data and are not stored
    g = arcgis_features(f"{CF}/CalVTP_treatement_areas_public_view/FeatureServer/0",
                        fields="Project_ID,Date_Completed,Treatment_Type,Treatment_Activity,Treatment_Acres,County,"
                               "Fuel_Type,Grant_Type,Status,Affiliation,TreatmentStage")
    n["treatments_calvtp"] = _save("treatments_calvtp", _clean(g))
    g = arcgis_features(f"{NIFC}/Integrated_Interagency_Fuels_Treatments_View/FeatureServer/0",
                        where="State='CA' OR N_ST_ABBR='CA'",
                        fields="TreatmentName,TreatmentType,TreatmentCategory,TreatmentStatus,TreatmentAcres,Agency,"
                               "UnitName,ActualInitiationDate,ActualCompletionDate,ActualCompletionFY,"
                               "TreatmentIdentifierDatabase,UniqueTreatmentID,FundingSource")
    n["treatments_federal"] = _save("treatments_federal", _clean(g))
    return n


def ingest_buildings(src=None, chunk=500_000):
    """Microsoft California footprints (one GeoJSON feature per line) -> 0.25-degree partitions."""
    from shapely.geometry import shape
    src = src or RAW.parent.parent / "ms_buildings" / "California.geojson"
    out = VECTORS / "buildings_ms"
    out.mkdir(parents=True, exist_ok=True)
    buf, total = {}, 0

    def flush():
        nonlocal buf
        for key, geoms in buf.items():
            g = gpd.GeoDataFrame(geometry=geoms, crs=4326)
            p = out / f"{key}.parquet"
            if p.exists():
                g = pd.concat([gpd.read_parquet(p), g], ignore_index=True)
            g.to_parquet(p, write_covering_bbox=True, row_group_size=20000)
        buf = {}

    with open(src) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{") or '"geometry"' not in line:
                continue
            try:
                geom = shape(json.loads(line)["geometry"])
            except Exception:  # noqa: BLE001
                continue
            c = geom.centroid
            key = f"{math.floor(c.x / PART)}_{math.floor(c.y / PART)}"
            buf.setdefault(key, []).append(geom)
            total += 1
            if total % chunk == 0:
                flush()
                log.info("buildings: %d", total)
    flush()
    # footprint area in m^2 (equal-area CRS), written once per partition
    for p in out.glob("*.parquet"):
        g = gpd.read_parquet(p)
        g["area_m2"] = g.to_crs(3310).area.round(1)
        g.to_parquet(p, write_covering_bbox=True, row_group_size=20000)
    (VECTORS / "buildings_ms.meta.json").write_text(json.dumps({"features": total, "partitions": len(list(
        out.glob("*.parquet"))), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return total


INGEST = {"fhsz": ingest_fhsz, "fire_history": ingest_fire_history, "ignitions": ingest_ignitions,
          "treatments": ingest_treatments, "buildings": ingest_buildings}


# ----------------------------------------------------------------------------- query
_load_lock = threading.Lock()


@lru_cache(maxsize=16)
def load(lid):
    """Whole layer in EPSG:4326 with its spatial index (not for buildings)."""
    spec = layer(lid)
    with _load_lock:
        g = gpd.read_parquet(spec["path"])
        g.sindex  # noqa: B018  build once
        return g


def query(lid, bbox, limit=None):
    """Features intersecting bbox (lon/lat), EPSG:4326."""
    w, s, e, n = bbox
    if lid == "buildings":
        parts = []
        d = VECTORS / "buildings_ms"
        for kx in range(math.floor(w / PART) - 1, math.floor(e / PART) + 1):
            for ky in range(math.floor(s / PART) - 1, math.floor(n / PART) + 1):
                p = d / f"{kx}_{ky}.parquet"
                if p.exists():
                    parts.append(gpd.read_parquet(p, bbox=(w, s, e, n)))
        g = pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(geometry=[], crs=4326)
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
        g = g[g.intersects(box(w, s, e, n))]
    else:
        g = load(lid)
        idx = g.sindex.query(box(w, s, e, n), predicate="intersects")
        g = g.iloc[np.sort(idx)]
    return g.head(limit) if limit else g


# ----------------------------------------------------------------------------- drawing into a web tile
def _rgba(hexcol, alpha):
    hexcol = hexcol.lstrip("#")
    return tuple(int(hexcol[k:k + 2], 16) for k in (0, 2, 4)) + (int(255 * alpha),)


def _color(spec, row):
    st = spec["style"]
    if "by" in st:
        return st["colors"].get(row.get(st["by"]), "#94a3b8")
    if "by_year" in st:
        y = row.get(st["by_year"])
        try:
            age = time.gmtime().tm_year - int(y)
        except (TypeError, ValueError):
            return st["fill"]
        return "#7e22ce" if age <= 10 else "#a855f7" if age <= 30 else "#c4b5fd"
    return st["fill"]


def draw(lid, z, x, y, size=256):
    """RGBA PIL image of the layer on web-mercator tile z/x/y, or None when nothing is there."""
    from PIL import Image, ImageDraw
    spec = layer(lid)
    if z < spec.get("min_zoom", 0):
        return None
    n = 2 ** z
    w = x / n * 360 - 180
    e = (x + 1) / n * 360 - 180
    nlat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    slat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    pad = (e - w) * 0.02
    g = query(lid, (w - pad, slat - pad, e + pad, nlat + pad))
    if g.empty:
        return None
    g = g.to_crs(3857)
    R = 6378137.0
    x0 = -math.pi * R + x * 2 * math.pi * R / n
    y0 = math.pi * R - y * 2 * math.pi * R / n
    px = 2 * math.pi * R / n / size
    if spec["geom"] == "polygon":
        g = g.assign(geometry=g.geometry.simplify(px * 0.75))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    st = spec["style"]
    alpha = st.get("alpha", 0.4)
    rows = g.drop(columns="geometry").to_dict("records")

    def pix(coords):
        a = np.asarray(coords)[:, :2]
        return list(map(tuple, np.column_stack(((a[:, 0] - x0) / px, (y0 - a[:, 1]) / px))))

    for geom, row in zip(g.geometry, rows):
        col = _color(spec, row)
        if spec["geom"] == "point":
            r = st.get("radius", 2) + (1 if z >= 12 else 0)
            cx, cy = (geom.x - x0) / px, (y0 - geom.y) / px
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_rgba(col, 0.9), outline=(15, 23, 42, 200))
            continue
        polys = getattr(geom, "geoms", [geom])
        for poly in polys:
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            ext = pix(poly.exterior.coords)
            if len(ext) < 3:
                continue
            dr.polygon(ext, fill=_rgba(col, alpha))
            for hole in poly.interiors:
                h = pix(hole.coords)
                if len(h) >= 3:
                    dr.polygon(h, fill=(0, 0, 0, 0))
            dr.line(ext + [ext[0]], fill=_rgba(st.get("line", col), 0.9), width=1)
    return img
