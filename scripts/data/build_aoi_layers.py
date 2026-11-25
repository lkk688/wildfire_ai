"""
Data layers for one area of interest (AOI), cut to a common UTM grid, for the map agent to compute on.

    python scripts/data/build_aoi_layers.py [--aoi san_bruno] [--only landfire,naip,...] [--force]

Writes /data/wildfire_benchmark_data/aoi/<aoi>/ (outside the repository):
    raster/<layer>.tif     GeoTIFF (tiled, deflate) in the AOI's UTM zone, snapped to the layer's own resolution
    vector/<layer>.gpkg    vector layers clipped to the AOI, in the same CRS
    partner/               Bellwether crops (partner data: never redistributed or published)
    raw/                   downloads kept as received
    manifest.json          one entry per layer: source, URL, date, resolution, units, licence, notes

Every layer is skipped if its file exists unless --force. Each fetcher is independent, so one failing source
does not block the others. All sources are public except Bellwether.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/data/wildfire_benchmark_data/aoi")
UA = {"User-Agent": "wildfire-ai-data-builder/1.0"}

AOIS = {
    # San Bruno city plus the wildland around it that a fire could come from (San Bruno Mountain, Sweeney Ridge,
    # Milagra Ridge, the Crystal Springs watershed) and the coast at Pacifica.
    "san_bruno": {"bbox": (-122.52, 37.54, -122.36, 37.70), "crs": "EPSG:32610"},
}

GTIFF = dict(driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")


# ----------------------------------------------------------------------------- grid helpers
class Grid:
    def __init__(self, bbox, crs, res):
        l, b, r, t = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
        self.left, self.bottom = np.floor(l / res) * res, np.floor(b / res) * res
        self.right, self.top = np.ceil(r / res) * res, np.ceil(t / res) * res
        self.res, self.crs = res, crs
        self.width = int(round((self.right - self.left) / res))
        self.height = int(round((self.top - self.bottom) / res))
        self.transform = from_origin(self.left, self.top, res, res)

    @property
    def bounds(self):
        return (self.left, self.bottom, self.right, self.top)

    def profile(self, count, dtype, nodata):
        return dict(GTIFF, width=self.width, height=self.height, count=count, dtype=dtype, crs=self.crs,
                    transform=self.transform, nodata=nodata)


def warp_to(src_paths, grid, out, dtype, nodata, resampling, band_names=None, bands=None):
    """Reproject one or more source rasters (same band layout) onto `grid`, later sources filling gaps."""
    srcs = [rasterio.open(p) for p in src_paths]
    count = len(bands) if bands else srcs[0].count
    bands = bands or list(range(1, count + 1))
    prof = grid.profile(count, dtype, nodata)
    with rasterio.open(out, "w", **prof) as dst:
        # windowed: one 4096-row strip at a time keeps memory flat for the 0.6 m imagery
        step = 4096
        for row0 in range(0, grid.height, step):
            h = min(step, grid.height - row0)
            win_tr = rasterio.windows.transform(rasterio.windows.Window(0, row0, grid.width, h), grid.transform)
            acc = np.full((count, h, grid.width), nodata, dtype=dtype)
            for s in srcs:
                for i, b in enumerate(bands):
                    tmp = np.full((h, grid.width), nodata, dtype=dtype)
                    reproject(rasterio.band(s, b), tmp, dst_transform=win_tr, dst_crs=grid.crs,
                              dst_nodata=nodata, resampling=resampling)
                    m = (acc[i] == nodata) & (tmp != nodata)
                    acc[i][m] = tmp[m]
            dst.write(acc, window=rasterio.windows.Window(0, row0, grid.width, h))
        if band_names:
            for i, n in enumerate(band_names, 1):
                dst.set_band_description(i, n)
    for s in srcs:
        s.close()
    return out


def stac():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def entry(layer_id, path, **kw):
    """Manifest entry. Metadata found while building (dates, scene ids) is kept in a sidecar <file>.meta.json so a
    later run that skips the layer still reports it."""
    p = Path(path)
    side = p.with_name(p.name + ".meta.json")
    e = json.load(open(side)) if side.exists() else {}
    e.update({k: v for k, v in kw.items() if v is not None})
    e.update(id=layer_id, path=str(p))
    e.setdefault("built_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if p.suffix == ".tif" and p.exists():
        with rasterio.open(p) as r:
            e.update(res_m=r.res[0], shape=[r.count, r.height, r.width], dtype=r.dtypes[0], nodata=r.nodata,
                     crs=str(r.crs), bands=[d or f"b{i+1}" for i, d in enumerate(r.descriptions)])
    if p.suffix == ".gpkg" and p.exists():
        import pyogrio
        e["features"] = int(pyogrio.read_info(p)["features"])
    if p.exists():
        json.dump(e, open(side, "w"), indent=1, default=str)
    return e


# ----------------------------------------------------------------------------- LANDFIRE (fuels, canopy, topography)
LF_LAYERS = ["LF2025_FBFM40", "LF2025_CC", "LF2025_CH", "LF2025_CBH", "LF2025_CBD", "LF2025_EVT", "LF2025_EVH",
             "LF2025_EVC", "LF2025_FDist", "LF2020_Elev", "LF2020_SlpD", "LF2020_Asp"]
# LANDFIRE's GeoServer WCS serves the raw CONUS rasters (EPSG:5070, 30 m) without sign-in. The LFPS job API now
# requires an e-mail address, which this script does not send.
LF_WCS = "https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/ows"


def landfire(aoi, d):
    out = d / "raster" / "landfire_30m.tif"
    if out.exists() and not FORCE:
        return entry("landfire", out, **LF_META)
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", "EPSG:5070", *AOIS[aoi]["bbox"], densify_pts=21)
    x0, y0, x1, y1 = x0 - 120, y0 - 120, x1 + 120, y1 + 120  # margin so the warp has full support at the edges
    raw = d / "raw" / "landfire"
    raw.mkdir(parents=True, exist_ok=True)
    grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 30)
    stack = []
    for lyr in LF_LAYERS:
        p = raw / f"{lyr}.tif"
        if not p.exists() or FORCE:
            r = requests.get(LF_WCS, headers=UA, timeout=300, params=[
                ("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
                ("coverageId", f"landfire_wcs__{lyr}_CONUS"), ("format", "image/tiff"),
                ("subset", f"X({x0},{x1})"), ("subset", f"Y({y0},{y1})")])
            r.raise_for_status()
            if not r.headers.get("content-type", "").startswith("image/tiff"):
                raise RuntimeError(f"{lyr}: {r.text[:200]}")
            p.write_bytes(r.content)
        tmp = raw / f"{lyr}_utm.tif"
        warp_to([p], grid, tmp, "int16", -9999, Resampling.nearest, band_names=[lyr])
        with rasterio.open(tmp) as t:
            stack.append(t.read(1))
        tmp.unlink()
    with rasterio.open(out, "w", **grid.profile(len(stack), "int16", -9999)) as dst:
        for i, (lyr, arr) in enumerate(zip(LF_LAYERS, stack), 1):
            dst.write(arr, i)
            dst.set_band_description(i, lyr)
    return entry("landfire", out, **LF_META)


LF_META = dict(kind="raster", source="LANDFIRE (USGS / USFS) via the LANDFIRE GeoServer WCS", url=LF_WCS,
               date="LF2025 fuels and vegetation (2024 disturbance); LF2020 topography", licence="Public domain",
               units={"FBFM40": "Scott & Burgan fuel model code", "CC": "% canopy cover", "CH": "m x 10",
                      "CBH": "m x 10", "CBD": "kg/m3 x 100", "EVT/EVH/EVC": "class codes (see LANDFIRE lookup)",
                      "Elev": "m", "SlpD": "degrees", "Asp": "degrees"},
               notes="Inputs to fire-spread models (FARSITE, FlamMap, ELMFIRE, Pyretechnics). Units as published by "
                     "LANDFIRE; check the layer's attribute table before use. Source values 32767 (fill) and "
                     "-9999 are both nodata; the source grid is EPSG:5070 30 m, warped nearest-neighbour.")


# ----------------------------------------------------------------------------- NAIP (0.6 m, RGB + near infrared)
def naip(aoi, d):
    out = d / "raster" / "naip_2022_0p6m.tif"
    meta = dict(kind="raster", source="USDA NAIP via Microsoft Planetary Computer", url="https://planetarycomputer.microsoft.com/dataset/naip",
                licence="Public domain", units="digital numbers 0-255", notes="Bands R, G, B, NIR. Leaf-on summer flight.")
    if out.exists() and not FORCE:
        return entry("naip", out, **meta)
    items = [i for i in stac().search(collections=["naip"], bbox=AOIS[aoi]["bbox"]).items()
             if i.properties["datetime"].startswith("2022")]
    if not items:
        raise RuntimeError("no 2022 NAIP items")
    grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 0.6)
    warp_to([i.assets["image"].href for i in items], grid, out, "uint8", 0, Resampling.bilinear,
            band_names=["red", "green", "blue", "nir"])
    dates = sorted({i.properties["datetime"][:10] for i in items})
    return entry("naip", out, date=f"{dates[0]} to {dates[-1]}", tiles=len(items), **meta)


# ----------------------------------------------------------------------------- 3DEP lidar (2 m)
def lidar(aoi, d):
    res = []
    for col, name, desc in [("3dep-lidar-hag", "lidar_hag_2m", "height above ground (vegetation and buildings), m"),
                            ("3dep-lidar-dtm", "lidar_dtm_2m", "bare-earth elevation, m"),
                            ("3dep-lidar-dsm", "lidar_dsm_2m", "surface elevation (top of canopy / roofs), m")]:
        out = d / "raster" / f"{name}.tif"
        meta = dict(kind="raster", source=f"USGS 3DEP lidar ({col}) via Microsoft Planetary Computer",
                    url=f"https://planetarycomputer.microsoft.com/dataset/{col}", licence="Public domain", units=desc)
        if not (out.exists() and not FORCE):
            items = list(stac().search(collections=[col], bbox=AOIS[aoi]["bbox"]).items())
            if not items:
                raise RuntimeError(f"no {col} items")
            grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 2)
            warp_to([i.assets["data"].href for i in items], grid, out, "float32", -9999.0, Resampling.bilinear,
                    band_names=[name])
            meta["date"] = ", ".join(sorted({(i.properties.get("start_datetime") or i.properties["datetime"])[:10]
                                             for i in items}))
            meta["tiles"] = len(items)
        res.append(entry(name, out, **meta))
    return res


# ----------------------------------------------------------------------------- Meta / WRI canopy height (1 m)
META_CHM = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float"


def _quadkey(lon, lat, z):
    import math
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return "".join(str((1 if x >> (i - 1) & 1 else 0) + (2 if y >> (i - 1) & 1 else 0)) for i in range(z, 0, -1))


def canopy_height(aoi, d):
    """3DEP lidar on Planetary Computer covers under 10% of this AOI, so vegetation height comes from Meta's
    global 1 m canopy height map (Tolan et al. 2024), which is a model of lidar from imagery, not a measurement."""
    out = d / "raster" / "canopy_height_meta_1m.tif"
    meta = dict(kind="raster", source="Meta / WRI High Resolution Canopy Height v6 (Tolan et al. 2024), AWS open data",
                url="https://registry.opendata.aws/dataforgood-fb-forests/", licence="CC BY 4.0",
                units="canopy height, m (0 = no canopy; 255 = outside source)",
                notes="Predicted from Maxar imagery by a model trained on aerial lidar; typical error a few metres. "
                      "Buildings are mostly 0 but tall structures can leak in.")
    if not (out.exists() and not FORCE):
        w, s, e, n = AOIS[aoi]["bbox"]
        keys = sorted({_quadkey(lo, la, 9) for lo in np.linspace(w, e, 5) for la in np.linspace(s, n, 5)})
        grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 1)
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            warp_to([f"{META_CHM}/chm/{k}.tif" for k in keys], grid, out, "uint8", 255, Resampling.bilinear,
                    band_names=["canopy_height_m"])
        # source-imagery dates: per-tile footprint GeoJSON with acq_date, area-weighted over the AOI
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from shapely.geometry import box, shape
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
        share, aoi_box = {}, box(w, s, e, n)
        for k in keys:
            fc = json.loads(s3.get_object(Bucket="dataforgood-fb-data", Key=f"forests/v1/alsgedi_global_v6_float/"
                                          f"metadata/{k}.geojson")["Body"].read())
            for f in fc["features"]:
                g = shape(f["geometry"])
                if g.intersects(aoi_box):
                    m = f["properties"]["acq_date"][:7]
                    share[m] = share.get(m, 0) + g.intersection(aoi_box).area
        tot = sum(share.values()) or 1
        meta["source_image_months"] = {m: round(v / tot, 3) for m, v in sorted(share.items())}
        meta["date"] = f"{min(share)} to {max(share)}" if share else None
        meta["tiles"] = keys
    return entry("canopy_height_meta", out, **meta)


# ----------------------------------------------------------------------------- 3DEP seamless DEM (10 m)
def dem(aoi, d):
    out = d / "raster" / "dem_10m.tif"
    meta = dict(kind="raster", source="USGS 3DEP seamless 1/3 arc-second DEM via Planetary Computer",
                url="https://planetarycomputer.microsoft.com/dataset/3dep-seamless", licence="Public domain", units="m")
    if not (out.exists() and not FORCE):
        items = [i for i in stac().search(collections=["3dep-seamless"], bbox=AOIS[aoi]["bbox"]).items()
                 if i.properties.get("gsd") == 10]
        grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 10)
        warp_to([i.assets["data"].href for i in items], grid, out, "float32", -9999.0, Resampling.bilinear,
                band_names=["elevation_m"])
    return entry("dem_10m", out, **meta)


# ----------------------------------------------------------------------------- ESA WorldCover (10 m)
def worldcover(aoi, d):
    out = d / "raster" / "worldcover_2021_10m.tif"
    meta = dict(kind="raster", source="ESA WorldCover 2021 v200 via Planetary Computer", licence="CC BY 4.0",
                url="https://planetarycomputer.microsoft.com/dataset/esa-worldcover",
                units="10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 70 snow, 80 water, 90 wetland, "
                      "95 mangrove, 100 moss")
    if not (out.exists() and not FORCE):
        items = [i for i in stac().search(collections=["esa-worldcover"], bbox=AOIS[aoi]["bbox"]).items()
                 if "2021" in i.properties.get("start_datetime", i.properties.get("datetime") or "")]
        grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 10)
        warp_to([i.assets["map"].href for i in items], grid, out, "uint8", 0, Resampling.nearest,
                band_names=["landcover"])
    return entry("worldcover", out, **meta)


# ----------------------------------------------------------------------------- Sentinel-2 (10 m): latest + dry season
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]


def sentinel2(aoi, d):
    res = []
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    dry_year = now.year if now.month >= 11 else now.year - 1
    windows = {"latest": f"{(now - timedelta(days=90)):%Y-%m-%d}/{now:%Y-%m-%d}",
               "dryseason": f"{dry_year}-09-01/{dry_year}-10-31"}
    for tag, rng in windows.items():
        out = d / "raster" / f"sentinel2_{tag}_10m.tif"
        meta = dict(kind="raster", source="Copernicus Sentinel-2 L2A via Planetary Computer", licence="Copernicus open",
                    url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
                    units="surface reflectance x 10000 (offset applied for processing baseline >= 04.00); SCL class",
                    notes="Single clearest scene in the window, not a composite. NDVI = (B08-B04)/(B08+B04); "
                          "NDMI (fuel moisture proxy) = (B08-B11)/(B08+B11).")
        if not (out.exists() and not FORCE):
            items = list(stac().search(collections=["sentinel-2-l2a"], bbox=AOIS[aoi]["bbox"], datetime=rng,
                                       query={"eo:cloud_cover": {"lt": 10}}).items())
            if not items:
                raise RuntimeError(f"no Sentinel-2 scene for {tag}")
            if tag == "latest":
                items.sort(key=lambda i: i.properties["datetime"], reverse=True)
            else:
                items.sort(key=lambda i: i.properties["eo:cloud_cover"])
            # one acquisition date (all tiles of it) so the layer is a single moment in time
            day = items[0].properties["datetime"][:10]
            same = [i for i in items if i.properties["datetime"][:10] == day]
            grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 10)
            tmp = []
            for bi, b in enumerate(S2_BANDS):
                p = d / "raw" / f"s2_{tag}_{b}.tif"
                p.parent.mkdir(parents=True, exist_ok=True)
                warp_to([i.assets[b].href for i in same], grid, p, "uint16", 0,
                        Resampling.nearest if b == "SCL" else Resampling.bilinear, band_names=[b])
                tmp.append(p)
            with rasterio.open(tmp[0]) as r0:
                prof = r0.profile
            prof.update(count=len(tmp))
            with rasterio.open(out, "w", **prof) as dst:
                for k, p in enumerate(tmp, 1):
                    with rasterio.open(p) as r:
                        dst.write(r.read(1), k)
                    dst.set_band_description(k, S2_BANDS[k - 1])
            for p in tmp:
                p.unlink()
            meta.update(date=day, scene_ids=[i.id for i in same],
                        cloud_cover=[i.properties["eo:cloud_cover"] for i in same])
        res.append(entry(f"sentinel2_{tag}", out, **meta))
    return res


# ----------------------------------------------------------------------------- USFS Wildfire Risk to Communities (30 m)
WRC_SVC = "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_{}/ImageServer"
WRC_LAYERS = {  # service name -> what it is (service stores integers; published value = service value / scale)
    "BurnProbability": ("annual burn probability", 10000),
    "ConditionalFlameLength": ("mean flame length if a fire occurs, ft", 10),
    "FlameLengthExceedProb4ft": ("probability flame length > 4 ft if a fire occurs", 1000),
    "FlameLengthExceedProb8ft": ("probability flame length > 8 ft if a fire occurs", 1000),
    "ExposureType": ("0 = not exposed ... 1 = directly exposed (indirect = embers)", 1000),
    "ConditionalRiskToPotentialStructures": ("consequence to a structure if a fire occurs, 0-100", 10),
    "RiskToPotentialStructures": ("BP x conditional risk to potential structures", None),
    "WildfireHazardPotential": ("wildfire hazard potential index", None),
}


def wrc(aoi, d):
    res = []
    grid = Grid(AOIS[aoi]["bbox"], AOIS[aoi]["crs"], 30)
    epsg = int(AOIS[aoi]["crs"].split(":")[1])
    for name, (desc, scale) in WRC_LAYERS.items():
        out = d / "raster" / f"wrc_{name}_30m.tif"
        if not (out.exists() and not FORCE):
            r = requests.get(WRC_SVC.format(name) + "/exportImage", headers=UA, timeout=120, params={
                "bbox": ",".join(map(str, grid.bounds)), "bboxSR": epsg, "imageSR": epsg,
                "size": f"{grid.width},{grid.height}", "format": "tiff", "pixelType": "U16",
                "interpolation": "RSP_NearestNeighbor", "f": "image"})
            r.raise_for_status()
            with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
                arr = s.read(1)
            with rasterio.open(out, "w", **grid.profile(1, "uint16", None)) as dst:
                dst.write(arr, 1)
                dst.set_band_description(1, name)
        res.append(entry(f"wrc_{name}", out, kind="raster", url=WRC_SVC.format(name),
                         source="USDA Forest Service, Wildfire Risk to Communities 2024 (RDS-2020-0016-2), "
                                "public image service", licence="Public domain", units=desc,
                         service_scale=scale,
                         notes="Raw integers from the display image service. The scale is inferred from the "
                               "service range and must be checked against the RDS source rasters before "
                               "quantitative use. Modelled from LANDFIRE fuels as of end-2020."))
    return res


# ----------------------------------------------------------------------------- vectors
def _arcgis_query(url, bbox, out_sr=4326):
    feats, offset = [], 0
    while True:
        r = requests.get(f"{url}/query", headers=UA, timeout=120, params={
            "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": out_sr, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 1000}).json()
        fs = r.get("features", [])
        feats += fs
        if len(fs) < 1000:
            break
        offset += 1000
    return {"type": "FeatureCollection", "features": feats}


def vectors(aoi, d):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon, box
    bbox, crs = AOIS[aoi]["bbox"], AOIS[aoi]["crs"]
    vd = d / "vector"
    vd.mkdir(parents=True, exist_ok=True)
    clip = gpd.GeoSeries([box(*bbox)], crs=4326)
    res = []

    def save(name, gdf, cut=False, **meta):
        p = vd / f"{name}.gpkg"
        gdf = gdf.to_crs(4326)
        if cut:  # large polygons (statewide zones, fire perimeters) are cut at the AOI edge
            gdf = gdf.assign(geometry=gdf.geometry.make_valid())
            gdf = gpd.clip(gdf, clip.iloc[0])
        else:  # buildings, parcels, streets: keep whole features that touch the AOI
            gdf = gdf[gdf.intersects(clip.iloc[0])]
        gdf.to_crs(crs).to_file(p, driver="GPKG")
        res.append(entry(name, p, kind="vector", features=len(gdf), **meta))

    def want(name):
        p = vd / f"{name}.gpkg"
        if p.exists() and not FORCE:
            res.append(entry(name, p, kind="vector", **VEC_META.get(name, {})))
            return False
        return True

    if want("buildings_ms"):
        df = pd.read_parquet(REPO / "data" / "ms_buildings_bayarea.parquet")
        w, s, e, n = bbox
        df = df[(df.lon >= w) & (df.lon <= e) & (df.lat >= s) & (df.lat <= n)]
        geom = [Polygon(json.loads(rg)) for rg in df.ring]
        save("buildings_ms", gpd.GeoDataFrame(df.drop(columns="ring"), geometry=geom, crs=4326),
             **VEC_META["buildings_ms"])
    smc = d / "raw" / "smc_county"
    for name, rel in [("buildings_county", "BUILDING_FOOTPRINTS/GIS_ADMIN.BUILDING_FOOTPRINTS.shp"),
                      ("parcels_county", "SAN_MATEO_COUNTY_ACTIVE_PARCELS/SAN_MATEO_COUNTY_ACTIVE_PARCELS.shp"),
                      ("streets_county", "SAN_MATEO_COUNTY_STREETS/SAN_MATEO_COUNTY_STREETS.shp")]:
        if (smc / rel).exists() and want(name):
            g = gpd.read_file(smc / rel, bbox=tuple(clip.to_crs(2227).total_bounds))
            save(name, g, **VEC_META[name])
    if want("fhsz"):
        parts = []
        for url, resp in [(SRA, "SRA (state)"), (LRA, "LRA (local)")]:
            g = gpd.GeoDataFrame.from_features(_arcgis_query(url, bbox)["features"], crs=4326)
            if len(g):
                g["responsibility_area"] = resp
                parts.append(g)
        save("fhsz", gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326), cut=True, **VEC_META["fhsz"])
    if want("fire_perimeters_historic"):
        g = gpd.GeoDataFrame.from_features(_arcgis_query(FRAP, bbox)["features"], crs=4326)
        save("fire_perimeters_historic", g, cut=True, **VEC_META["fire_perimeters_historic"])
    return res


SRA = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/FHSZSRA_23_3/FeatureServer/0"
LRA = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/FHSALRA25_v1_All/FeatureServer/0"
FRAP = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Historic_Fire_Perimeters/"
        "FeatureServer/2")
VEC_META = {
    "buildings_ms": dict(source="Microsoft US Building Footprints (local extract)", licence="ODbL",
                         notes="Footprints only, imagery mostly 2018-2020; no height or attributes."),
    "buildings_county": dict(source="San Mateo County GIS, BUILDING_FOOTPRINTS (exported daily)",
                             url="https://www.smcgov.org/isd/gis-data-download", licence="County open data",
                             notes="Has BASE_ELEV and BUILDING_E (roof elevation, ft) so height = BUILDING_E - "
                                   "BASE_ELEV; no year built."),
    "parcels_county": dict(source="San Mateo County GIS, active parcels (exported daily)",
                           url="https://www.smcgov.org/isd/gis-data-download", licence="County open data",
                           notes="APN and situs address only; year built / structure type are in the Assessor roll, "
                                 "which is not in this export."),
    "streets_county": dict(source="San Mateo County GIS, street centrelines (NG911)",
                           url="https://www.smcgov.org/isd/gis-data-download", licence="County open data",
                           notes="Road class and speed; useful for access and as potential fuel breaks."),
    "fhsz": dict(source="CAL FIRE Fire Hazard Severity Zones (SRA 2023, LRA 2025)", url=SRA, licence="Public"),
    "fire_perimeters_historic": dict(source="CAL FIRE FRAP historical fire perimeters (1950+)", url=FRAP,
                                     licence="Public"),
}


# ----------------------------------------------------------------------------- Bellwether (partner, never shared)
def bellwether(aoi, d):
    src = REPO / "SBFire_SJSU-20260730T030001Z-1-001" / "SBFire_SJSU" / "ref_data"
    pd_ = d / "partner"
    pd_.mkdir(exist_ok=True)
    res = []
    w, s, e, n = AOIS[aoi]["bbox"]
    for f in ["prediction_for_20260401_to_20270401.tif", "prediction_for_20260401_to_20310401.tif",
              "risk_factors_1_year_cog.tif", "risk_factors_5_year_cog.tif"]:
        out = pd_ / f
        if not (out.exists() and not FORCE):
            with rasterio.open(src / f) as r:
                win = rasterio.windows.from_bounds(w, s, e, n, r.transform).round_offsets().round_lengths()
                prof = dict(r.profile, **GTIFF, width=int(win.width), height=int(win.height),
                            transform=r.window_transform(win))
                with rasterio.open(out, "w", **prof) as dst:
                    dst.write(r.read(window=win, boundless=True, fill_value=r.nodata or 0))
                    for i, desc in enumerate(r.descriptions, 1):
                        if desc:
                            dst.set_band_description(i, desc)
        res.append(entry(f"bellwether_{f.split('.')[0]}", out, kind="raster",
                         source="Bellwether (X, Alphabet), partner release", licence="PARTNER DATA - do not "
                         "redistribute, publish or show", notes="Native grid kept (EPSG:4326, ~100 m)."))
    return res


FETCHERS = {"landfire": landfire, "naip": naip, "lidar": lidar, "chm": canopy_height, "dem": dem, "worldcover": worldcover,
            "sentinel2": sentinel2, "wrc": wrc, "vectors": vectors, "bellwether": bellwether}
FORCE = False


def main():
    global FORCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="san_bruno")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    FORCE = a.force
    d = ROOT / a.aoi
    for sub in ("raster", "vector", "raw"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    mpath = d / "manifest.json"
    manifest = json.load(open(mpath)) if mpath.exists() else {"aoi": a.aoi, "layers": {}}
    manifest.update(bbox_wgs84=AOIS[a.aoi]["bbox"], crs=AOIS[a.aoi]["crs"])
    only = [x for x in a.only.split(",") if x] or list(FETCHERS)
    for name in only:
        t0 = time.time()
        try:
            out = FETCHERS[name](a.aoi, d)
            for e in (out if isinstance(out, list) else [out]):
                manifest["layers"][e["id"]] = e
            print(f"[ok] {name} ({time.time() - t0:.0f} s)")
            manifest.get("errors", {}).pop(name, None)
        except Exception as ex:  # noqa: BLE001
            manifest.setdefault("errors", {})[name] = f"{type(ex).__name__}: {ex}"
            print(f"[FAIL] {name}: {type(ex).__name__}: {ex}")
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        json.dump(manifest, open(mpath, "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
