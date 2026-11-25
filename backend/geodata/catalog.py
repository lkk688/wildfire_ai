"""
Every layer the cache can serve: where it comes from, its units, how raw values decode, and how the map draws it.

Raster spec keys
    kind        continuous | categorical | rgb
    dtype/nodata storage type and nodata of the cached tiles
    scale       decoded value = raw * scale (None: raw value is the value)
    res         native resolution (m) of pyramid level 0; level L has res * 2**L
    max_level   coarsest level kept; zoomed further out the layer is not drawn
    prefetch    True: the whole state is downloaded by `build prefetch`; False: tiles are fetched when first viewed
    src         fetcher name (sources.py) and its parameters
    style       continuous: colormap, vmin, vmax (decoded units), below (transparent under this value)
                categorical: {code: [label, "#rrggbb"]}
"""
from datetime import date

from backend.geodata.config import VECTORS

_today = date.today()
DRY_YEAR = _today.year if _today.month >= 11 else _today.year - 1  # latest finished Sep-Oct dry season

LF = "LANDFIRE (USGS / USFS), via the LANDFIRE GeoServer WCS"
LF_URL = "https://landfire.gov"
WRC = "USDA Forest Service, Wildfire Risk to Communities 2024 (RDS-2020-0016-2), public image service"
WRC_URL = "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_{}/ImageServer"
WRC_NOTE = ("Modelled with FSim on LANDFIRE fuels as of end-2020. Values are the integers of the public display "
            "service decoded with the stated scale; check against the RDS source rasters before publishing numbers.")

FBFM40 = {
    91: ["NB1 urban / developed", "#9ca3af"], 92: ["NB2 snow / ice", "#e0f2fe"], 93: ["NB3 agriculture", "#d9f99d"],
    98: ["NB8 open water", "#93c5fd"], 99: ["NB9 bare ground", "#e7e5e4"],
    **{c: [f"GR{c - 100} grass", col] for c, col in zip(range(101, 110), ["#fef9c3", "#fef08a", "#fde047", "#facc15",
                                                                        "#eab308", "#ca8a04", "#a16207", "#854d0e",
                                                                        "#713f12"])},
    **{c: [f"GS{c - 120} grass-shrub", col] for c, col in zip(range(121, 125), ["#fed7aa", "#fdba74", "#fb923c",
                                                                              "#f97316"])},
    **{c: [f"SH{c - 140} shrub", col] for c, col in zip(range(141, 150), ["#fecaca", "#fca5a5", "#f87171", "#ef4444",
                                                                        "#dc2626", "#b91c1c", "#991b1b", "#7f1d1d",
                                                                        "#450a0a"])},
    **{c: [f"TU{c - 160} timber-understory", col] for c, col in zip(range(161, 166), ["#bbf7d0", "#86efac", "#4ade80",
                                                                                    "#22c55e", "#16a34a"])},
    **{c: [f"TL{c - 180} timber litter", col] for c, col in zip(range(181, 190), ["#a7f3d0", "#6ee7b7", "#34d399",
                                                                                "#10b981", "#059669", "#047857",
                                                                                "#065f46", "#064e3b", "#022c22"])},
    **{c: [f"SB{c - 200} slash-blowdown", col] for c, col in zip(range(201, 205), ["#e9d5ff", "#c084fc", "#9333ea",
                                                                                 "#6b21a8"])},
}
WORLDCOVER = {10: ["Tree cover", "#006400"], 20: ["Shrubland", "#ffbb22"], 30: ["Grassland", "#ffff4c"],
              40: ["Cropland", "#f096ff"], 50: ["Built-up", "#fa0000"], 60: ["Bare / sparse", "#b4b4b4"],
              70: ["Snow and ice", "#f0f0f0"], 80: ["Water", "#0064c8"], 90: ["Herbaceous wetland", "#0096a0"],
              95: ["Mangroves", "#00cf75"], 100: ["Moss and lichen", "#fae6a0"]}


def _lf(cov, title, units, scale, style, kind="continuous", note=""):
    return dict(title=title, group="Fuels and terrain (LANDFIRE, 30 m)", kind=kind, dtype="int16", nodata=-9999,
                scale=scale, res=30, max_level=7, prefetch=True, units=units, source=LF, url=LF_URL,
                date="LF2025 (2024 disturbances)" if cov.startswith("LF2025") else "LF2020",
                licence="Public domain", src={"fetch": "landfire_wcs", "coverage": cov}, resampling="nearest",
                style=style, notes=("Input to fire-spread models (FARSITE, FlamMap, ELMFIRE, Pyretechnics). " + note).strip())


def _wrc(svc, title, units, scale, style):
    return dict(title=title, group="Modelled wildfire hazard (USFS WRC, 30 m)", kind="continuous", dtype="uint16",
                nodata=0, scale=scale, res=30, max_level=7, prefetch=True, units=units, source=WRC,
                url=WRC_URL.format(svc), date="2024 release; landscape end-2020", licence="Public domain",
                src={"fetch": "arcgis_export", "url": WRC_URL.format(svc)}, resampling="nearest", style=style,
                notes=WRC_NOTE)


FBFM40_GROUPS = [["Grass (GR1-9)", "#facc15"], ["Grass-shrub (GS1-4)", "#fb923c"], ["Shrub (SH1-9)", "#dc2626"],
                 ["Timber understory (TU1-5)", "#22c55e"], ["Timber litter (TL1-9)", "#059669"],
                 ["Slash / blowdown (SB1-4)", "#9333ea"], ["Urban (NB1)", "#9ca3af"], ["Agriculture (NB3)", "#d9f99d"],
                 ["Water (NB8)", "#93c5fd"], ["Bare (NB9)", "#e7e5e4"]]

RASTERS = {
    # ---- imagery and vegetation structure
    "naip": dict(title="Aerial photo (NAIP, 0.6 m, RGB + near-infrared)", group="Imagery and vegetation",
                 kind="rgb", dtype="uint8", nodata=0, bands_n=4, scale=None, res=0.6, max_level=5, prefetch=False,
                 units="digital number 0-255 (bands red, green, blue, NIR)",
                 source="USDA NAIP via Microsoft Planetary Computer",
                 url="https://planetarycomputer.microsoft.com/dataset/naip", date="newest flight per place (CA: 2022)",
                 licence="Public domain", src={"fetch": "stac_cog", "collection": "naip", "asset": "image",
                                              "bands": [1, 2, 3, 4]},
                 resampling="bilinear", style={"bands": [0, 1, 2]},
                 notes="Fetched when first viewed. Leaf-on flight; the newest year wins where flights overlap."),
    "canopy_height": dict(title="Tree canopy height (Meta / WRI, 1 m)", group="Imagery and vegetation",
                          kind="continuous", dtype="uint8", nodata=255, scale=None, res=1.0, max_level=1,
                          prefetch=False, units="m", source="Meta / WRI High Resolution Canopy Height v6 "
                          "(Tolan et al. 2024), AWS open data", url="https://registry.opendata.aws/dataforgood-fb-forests/",
                          date="model of Maxar imagery, mostly 2018-2020", licence="CC BY 4.0",
                          src={"fetch": "meta_chm"}, resampling="bilinear",
                          style={"cmap": "Greens", "vmin": 0, "vmax": 30, "below": 1},
                          notes="Predicted from satellite imagery by a model trained on aerial lidar (typical error a "
                                "few metres), not a measurement. The source has no overviews, so it is shown only "
                                "when zoomed in."),
    "worldcover": dict(title="Land cover (ESA WorldCover 2021, 10 m)", group="Imagery and vegetation",
                       kind="categorical", dtype="uint8", nodata=0, scale=None, res=10, max_level=8, prefetch=True,
                       units="class", source="ESA WorldCover 2021 v200 via Planetary Computer",
                       url="https://planetarycomputer.microsoft.com/dataset/esa-worldcover", date="2021",
                       licence="CC BY 4.0", src={"fetch": "stac_cog", "collection": "esa-worldcover", "asset": "map",
                                                "filter": lambda i: "2021" in (i.properties.get("start_datetime")
                                                                               or i.properties.get("datetime") or "")},
                       resampling="mode", style=WORLDCOVER),
    "ndvi_dry": dict(title=f"Vegetation greenness, dry season {DRY_YEAR} (Sentinel-2 NDVI, 10 m)",
                     group="Imagery and vegetation", kind="continuous", dtype="int16", nodata=-32768, scale=1e-4,
                     res=10, max_level=6, prefetch=False, units="NDVI (-1 to 1)",
                     source="Copernicus Sentinel-2 L2A via Planetary Computer",
                     url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
                     date=f"clearest scenes {DRY_YEAR}-09-01 to {DRY_YEAR}-10-31", licence="Copernicus open",
                     src={"fetch": "s2_index", "a": "B08", "b": "B04", "window": f"{DRY_YEAR}-09-01/{DRY_YEAR}-10-31"},
                     style={"cmap": "YlGn", "vmin": 0, "vmax": 0.8, "below": -0.05},
                     notes="Per tile the clearest scenes in the window; cloud, shadow and snow masked with the scene "
                           "classification. Adjacent tiles can come from different days."),
    "ndmi_dry": dict(title=f"Vegetation moisture, dry season {DRY_YEAR} (Sentinel-2 NDMI, 10-20 m)",
                     group="Imagery and vegetation", kind="continuous", dtype="int16", nodata=-32768, scale=1e-4,
                     res=10, max_level=6, prefetch=False, units="NDMI (-1 to 1); lower = drier",
                     source="Copernicus Sentinel-2 L2A via Planetary Computer",
                     url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
                     date=f"clearest scenes {DRY_YEAR}-09-01 to {DRY_YEAR}-10-31", licence="Copernicus open",
                     src={"fetch": "s2_index", "a": "B08", "b": "B11", "window": f"{DRY_YEAR}-09-01/{DRY_YEAR}-10-31"},
                     style={"cmap": "BrBG", "vmin": -0.3, "vmax": 0.4},
                     notes="(NIR - SWIR) / (NIR + SWIR), a proxy for live fuel moisture. Same scene rules as NDVI."),
    # ---- terrain
    "dem": dict(title="Elevation (USGS 3DEP, 10 m)", group="Fuels and terrain (LANDFIRE, 30 m)", kind="continuous",
                dtype="float32", nodata=-9999.0, scale=None, res=10, max_level=8, prefetch=True, units="m",
                source="USGS 3DEP 1/3 arc-second DEM via Planetary Computer",
                url="https://planetarycomputer.microsoft.com/dataset/3dep-seamless", date="3DEP seamless",
                licence="Public domain", src={"fetch": "stac_cog", "collection": "3dep-seamless", "asset": "data",
                                             "filter": lambda i: i.properties.get("gsd") == 10},
                resampling="bilinear", style={"cmap": "terrain", "vmin": 0, "vmax": 3000}),
    # ---- LANDFIRE fuels and topography (spread-model inputs)
    "fuel_model": _lf("LF2025_FBFM40", "Fire behaviour fuel model (Scott & Burgan 40)", "fuel model code", None,
                      FBFM40, kind="categorical"),
    "canopy_cover": _lf("LF2025_CC", "Forest canopy cover", "%", None,
                        {"cmap": "Greens", "vmin": 0, "vmax": 90, "below": 1}),
    "canopy_height_lf": _lf("LF2025_CH", "Forest canopy height", "m", 0.1,
                            {"cmap": "Greens", "vmin": 0, "vmax": 40, "below": 0.1}),
    "canopy_base_height": _lf("LF2025_CBH", "Canopy base height", "m", 0.1,
                              {"cmap": "YlOrBr", "vmin": 0, "vmax": 8, "below": 0.1}),
    "canopy_bulk_density": _lf("LF2025_CBD", "Canopy bulk density", "kg/m3", 0.01,
                               {"cmap": "OrRd", "vmin": 0, "vmax": 0.3, "below": 0.005}),
    "slope": _lf("LF2020_SlpD", "Slope", "degrees", None, {"cmap": "magma", "vmin": 0, "vmax": 45, "below": 0.5}),
    "aspect": _lf("LF2020_Asp", "Aspect (direction the slope faces)", "degrees from north (-1 flat)", None,
                  {"cmap": "twilight", "vmin": 0, "vmax": 360, "below": 0}),
    # ---- USFS Wildfire Risk to Communities
    "burn_probability": _wrc("BurnProbability", "Annual burn probability", "probability per year", 1e-4,
                             {"cmap": "OrRd", "vmin": 0, "vmax": 0.02, "below": 0.0001}),
    "flame_length": _wrc("ConditionalFlameLength", "Flame length if a fire occurs", "ft", 0.1,
                         {"cmap": "YlOrRd", "vmin": 0, "vmax": 20, "below": 0.1}),
    "flame_gt4ft": _wrc("FlameLengthExceedProb4ft", "Chance flames exceed 4 ft if a fire occurs", "probability",
                        1e-3, {"cmap": "YlOrRd", "vmin": 0, "vmax": 1, "below": 0.01}),
    "flame_gt8ft": _wrc("FlameLengthExceedProb8ft", "Chance flames exceed 8 ft if a fire occurs", "probability",
                        1e-3, {"cmap": "YlOrRd", "vmin": 0, "vmax": 1, "below": 0.01}),
    "exposure_type": _wrc("ExposureType", "Exposure type (1 direct flame, <1 embers only, 0 not exposed)",
                          "0-1", 1e-3, {"cmap": "PuRd", "vmin": 0, "vmax": 1, "below": 0.01}),
    "cond_risk_structures": _wrc("ConditionalRiskToPotentialStructures",
                                 "Damage to a home if a fire occurs (conditional risk)", "index 0-100", 0.1,
                                 {"cmap": "Reds", "vmin": 0, "vmax": 100, "below": 1}),
    "risk_structures": _wrc("RiskToPotentialStructures", "Risk to potential structures", "index", None,
                            {"cmap": "Reds", "vmin": 0, "vmax": 200, "below": 1}),
    "hazard_potential": _wrc("WildfireHazardPotential", "Wildfire hazard potential", "index", None,
                             {"cmap": "YlOrRd", "vmin": 0, "vmax": 5000, "below": 1}),
}

RASTERS["fuel_model"]["legend_swatches"] = FBFM40_GROUPS

# ---- fire weather from gridMET (built by build.py into derived/fireweather_gridmet.tif)
GM = dict(group="Fire weather 2000-2025 (gridMET, 4 km)", kind="continuous", dtype="float32", nodata=-9999.0,
          scale=None, res=500, max_level=2, prefetch=True, source="gridMET (Abatzoglou 2013), University of Idaho",
          url="https://www.climatologylab.org/gridmet.html", date="daily 2000-2025", licence="Public domain",
          resampling="nearest",
          notes="Fire-weather days = days when the energy release component (ERC, fuel model G) is at or above that "
                "cell's 90th percentile of 2000-2025. Winds are daily means at 10 m, so gusts are higher.")
FW_BANDS = [  # band order in derived/fireweather_gridmet.tif
    ("fw_erc_p90", "ERC, 90th percentile (fire-weather threshold)", "ERC", {"cmap": "YlOrRd", "vmin": 20, "vmax": 100}),
    ("fw_erc_p97", "ERC, 97th percentile", "ERC", {"cmap": "YlOrRd", "vmin": 20, "vmax": 110}),
    ("fw_wind_p50", "Wind speed on fire-weather days, median", "m/s", {"cmap": "PuBu", "vmin": 0, "vmax": 8}),
    ("fw_wind_p90", "Wind speed on fire-weather days, 90th percentile", "m/s", {"cmap": "PuBu", "vmin": 0, "vmax": 12}),
    ("fw_wind_max", "Wind speed on fire-weather days, maximum", "m/s", {"cmap": "PuBu", "vmin": 0, "vmax": 20}),
    ("fw_dir_mode", "Most common wind direction on fire-weather days (from)", "degrees from north",
     {"cmap": "twilight", "vmin": 0, "vmax": 360}),
    ("fw_dir_mode_share", "Share of fire-weather days from that direction", "fraction",
     {"cmap": "BuPu", "vmin": 0.1, "vmax": 0.7}),
    ("fw_dir_strong", "Wind direction on the windiest 10% of fire-weather days (from)", "degrees from north",
     {"cmap": "twilight", "vmin": 0, "vmax": 360}),
    ("fw_rmin_p10", "Minimum humidity on fire-weather days, 10th percentile", "%",
     {"cmap": "YlOrBr_r", "vmin": 3, "vmax": 30}),
    ("fw_fm100_p10", "100-hour dead fuel moisture on fire-weather days, 10th percentile", "%",
     {"cmap": "YlOrBr_r", "vmin": 2, "vmax": 12}),
]
for _k, (_id, _t, _u, _st) in enumerate(FW_BANDS, 1):
    RASTERS[_id] = dict(GM, title=_t, units=_u, style=_st,
                        src={"fetch": "local_file", "file": "fireweather_gridmet.tif", "band": _k})

# ---- vectors: statewide GeoParquet in VECTORS (built by build.py)
VECTOR_LAYERS = {
    "buildings": dict(title="Building footprints (Microsoft)", group="Buildings and places", geom="polygon",
                      file="buildings_ms.parquet", min_zoom=15, source="Microsoft US Building Footprints",
                      url="https://github.com/microsoft/USBuildingFootprints", date="imagery mostly 2018-2020",
                      licence="ODbL", style={"fill": "#e2e8f0", "line": "#0f172a", "alpha": 0.55},
                      notes="Footprints only. San Mateo County footprints with heights are in the San Bruno AOI."),
    "fhsz": dict(title="Fire Hazard Severity Zones (CAL FIRE)", group="Official hazard and history", geom="polygon",
                 file="fhsz.parquet", min_zoom=6, source="CAL FIRE FHSZ, SRA 2023 and LRA 2025",
                 url="https://osfm.fire.ca.gov/what-we-do/community-wildfire-preparedness-and-mitigation/fire-hazard-severity-zones",
                 date="SRA 2023, LRA 2025", licence="Public", label="FHSZ_Description",
                 style={"by": "FHSZ_Description", "colors": {"Moderate": "#facc15", "High": "#f97316",
                                                            "Very High": "#dc2626"}, "alpha": 0.35}),
    "fire_history": dict(title="Fire perimeters 1950-2025 (CAL FIRE FRAP)", group="Official hazard and history",
                         geom="polygon", file="fire_history.parquet", min_zoom=6,
                         source="CAL FIRE FRAP historical fire perimeters", label="FIRE_NAME",
                         url="https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters",
                         date="through 2025", licence="Public",
                         style={"by_year": "YEAR_", "fill": "#a855f7", "line": "#7e22ce", "alpha": 0.25}),
    "ignitions": dict(title="Wildfire ignition points 1992-2026 (NIFC InFORM)", group="Official hazard and history",
                      geom="point", file="ignitions.parquet", min_zoom=8,
                      source="NIFC InFORM fire occurrence (FPA-FOD lineage), California points",
                      url="https://data-nifc.opendata.arcgis.com", date="1992 to present", licence="Public",
                      label="IncidentName",
                      style={"by": "cause", "colors": {"Human": "#f97316", "Natural": "#3b82f6",
                                                       "Undetermined": "#94a3b8"}, "radius": 2},
                      notes="Point of origin and discovery date of each reported wildfire, with cause and final size. "
                            "Federal and state agencies plus local departments that report to InFORM."),
    "treatments_calfire": dict(title="CAL FIRE fuel-reduction projects (CalMAPPER / FTER)",
                               group="Fuel treatments", geom="polygon", file="treatments_calfire.parquet", min_zoom=8,
                               source="CAL FIRE Fuels Treatment Effectiveness Reporting, treatment polygons",
                               url="https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/FTER%20Treatment%20Polygons%20Public%20View/FeatureServer/0",
                               date="through present", licence="Public", label="TREATMENT_NAME",
                               style={"fill": "#10b981", "line": "#047857", "alpha": 0.35}),
    "treatments_rx": dict(title="Prescribed fire perimeters (CAL FIRE)", group="Fuel treatments", geom="polygon",
                          file="treatments_rx.parquet", min_zoom=8, source="CAL FIRE prescribed fire perimeters",
                          url="https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Prescribed_Fire_Perimeters_Public/FeatureServer/0",
                          date="through present", licence="Public", label="TREATMENT_NAME",
                          style={"fill": "#22d3ee", "line": "#0e7490", "alpha": 0.35}),
    "treatments_calvtp": dict(title="CalVTP vegetation treatment areas", group="Fuel treatments", geom="polygon",
                              file="treatments_calvtp.parquet", min_zoom=8,
                              source="Board of Forestry CalVTP treatment areas (contact fields dropped)",
                              url="https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/CalVTP_treatement_areas_public_view/FeatureServer/0",
                              date="through present", licence="Public", label="Treatment_Type",
                              style={"fill": "#84cc16", "line": "#4d7c0f", "alpha": 0.35}),
    "treatments_federal": dict(title="Federal fuel treatments (interagency, USFS / DOI)", group="Fuel treatments",
                               geom="polygon", file="treatments_federal.parquet", min_zoom=8,
                               source="NIFC Integrated Interagency Fuels Treatments (NFPORS + USFS FACTS)",
                               url="https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Integrated_Interagency_Fuels_Treatments_View/FeatureServer/0",
                               date="through present", licence="Public", label="TreatmentName",
                               style={"fill": "#14b8a6", "line": "#0f766e", "alpha": 0.3}),
}


def layer(layer_id):
    if layer_id in RASTERS:
        return dict(RASTERS[layer_id], id=layer_id, type="raster")
    if layer_id in VECTOR_LAYERS:
        return dict(VECTOR_LAYERS[layer_id], id=layer_id, type="vector", path=str(VECTORS / VECTOR_LAYERS[layer_id]["file"]))
    raise KeyError(f"unknown layer {layer_id!r}")


def _public(spec):
    keep = ("id", "type", "title", "group", "kind", "units", "scale", "res", "source", "url", "date", "licence",
            "notes", "style", "geom", "min_zoom", "prefetch", "label")
    out = {k: spec[k] for k in keep if k in spec}
    out["legend"] = legend(spec)
    if spec["type"] == "raster":
        out["min_zoom"] = min_zoom(spec)
        out["max_native_zoom"] = 19
    return out


def legend(spec):
    """What the map legend shows: a colour ramp (continuous) or labelled swatches (categorical, vectors)."""
    st = spec.get("style", {})
    if spec.get("legend_swatches"):
        return {"swatches": spec["legend_swatches"]}
    if spec["type"] == "vector":
        if "colors" in st:
            return {"swatches": [[k, v] for k, v in st["colors"].items()], "shape": "dot" if spec["geom"] == "point" else "box"}
        if "by_year" in st:
            return {"swatches": [["last 10 years", "#7e22ce"], ["10-30 years ago", "#a855f7"], ["older", "#c4b5fd"]]}
        return {"swatches": [[spec["title"].split(" (")[0], st.get("fill", "#94a3b8")]]}
    if spec["kind"] == "categorical":
        return {"swatches": [[lab, col] for _, (lab, col) in sorted(st.items())]}
    if spec["kind"] == "rgb":
        return None
    import matplotlib
    cm = matplotlib.colormaps[st["cmap"]]
    stops = ["#%02x%02x%02x" % tuple(int(255 * c) for c in cm(t)[:3]) for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    return {"ramp": stops, "min": st["vmin"], "max": st["vmax"], "units": spec.get("units")}


def min_zoom(spec, lat=37.0):
    """Web zoom below which the raster is not drawn (its coarsest pyramid level would be too fine)."""
    import math
    coarsest = spec["res"] * 2 ** spec["max_level"]
    return max(0, math.ceil(math.log2(156543.03 * math.cos(math.radians(lat)) / (coarsest * 2))))


def catalog():
    """Public description of every layer (what the map and the agent see), with whether its data is built."""
    from backend.geodata.config import TILES
    out = []
    for lid in list(RASTERS) + list(VECTOR_LAYERS):
        spec = layer(lid)
        p = _public(spec)
        if spec["type"] == "raster":
            p["cached_tiles_L0"] = sum(1 for _ in (TILES / lid / "L0").glob("*.tif")) if (TILES / lid / "L0").exists() else 0
            if spec["src"]["fetch"] == "local_file":
                from backend.geodata.config import DERIVED
                p["available"] = (DERIVED / spec["src"]["file"]).exists()
            else:
                p["available"] = True
        else:
            from pathlib import Path
            p["available"] = Path(spec["path"]).exists()
        out.append(p)
    return out
