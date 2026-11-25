# California data cache

The map's live layers answer "what is happening now". Planning questions (where would a fire run, which vegetation
matters, where to clear first) need more: fuel models, canopy structure, terrain, fire-weather history, past
ignitions and past treatments, for the whole state, readable in milliseconds by the map and by the AI agent.
`backend/geodata/` keeps these layers on our own disk.

## How it works

- **Rasters** are stored as a tile pyramid in California Albers (EPSG:3310), 512 × 512 pixels per tile. Level 0 is
  the layer's native resolution; each level above halves it. A tile is read from disk if present; otherwise it is
  fetched from the public source, written to disk, and returned. Nothing outside California is fetched.
- **Statewide layers** (marked *statewide* below) are downloaded in full ahead of time and their coarse levels are
  built from level 0, so the zoomed-out map and the full-resolution queries agree.
- **On-view layers** (aerial photos, canopy height, Sentinel-2) are too large to copy for the whole state; each tile
  is fetched the first time someone looks at or asks about that place, then kept.
- **Vectors** are statewide GeoParquet files (EPSG:4326). Building footprints are split into 0.25° partitions.
- **Map tiles** (`/api/geodata/tiles/<layer>/<z>/<x>/<y>.png`) are drawn from the cache and kept as PNG.

Default location: `/data/wildfire_benchmark_data/ca` (set `GEODATA_ROOT` to change it).

## Layers

| Layer id | What | Source | Resolution | Cache |
|---|---|---|---|---|
| `fuel_model` | Scott & Burgan 40 fire-behaviour fuel model | LANDFIRE LF2025 | 30 m | statewide |
| `canopy_cover`, `canopy_height_lf`, `canopy_base_height`, `canopy_bulk_density` | Forest canopy structure (crown-fire inputs) | LANDFIRE LF2025 | 30 m | statewide |
| `slope`, `aspect` | Terrain | LANDFIRE LF2020 | 30 m | statewide |
| `dem` | Elevation | USGS 3DEP 1/3 arc-second | 10 m | statewide |
| `worldcover` | Land cover (tree, shrub, grass, built-up, ...) | ESA WorldCover 2021 | 10 m | statewide |
| `burn_probability`, `flame_length`, `flame_gt4ft`, `flame_gt8ft`, `exposure_type`, `cond_risk_structures`, `risk_structures`, `hazard_potential` | Modelled wildfire likelihood and intensity, and the exposure of homes | USFS Wildfire Risk to Communities 2024 | 30 m | statewide |
| `fw_erc_p90` … `fw_fm100_p10` (10 layers) | Fire-weather climatology 2000–2025: ERC thresholds, wind speed and direction on fire-weather days, low humidity, 100-h fuel moisture | gridMET, computed here | 4 km | statewide |
| `naip` | Aerial photo, red / green / blue / near-infrared | USDA NAIP (2022 in California) | 0.6 m | on view |
| `canopy_height` | Tree canopy height | Meta / WRI model (imagery 2018–20) | 1 m | on view |
| `ndvi_dry`, `ndmi_dry` | Vegetation greenness and moisture in the latest Sep–Oct dry season | Sentinel-2 L2A | 10 m | on view |
| `fhsz` | Fire Hazard Severity Zones (Moderate, High, Very High) | CAL FIRE (SRA 2023, LRA 2025) | polygons | statewide |
| `fire_history` | Fire perimeters since 1950 | CAL FIRE FRAP | polygons | statewide |
| `ignitions` | Points of origin of 183,000 wildfires since 1992, with cause, date and size | NIFC InFORM (FPA-FOD lineage) | points | statewide |
| `treatments_calfire`, `treatments_rx`, `treatments_calvtp`, `treatments_federal` | Fuel-reduction projects, prescribed burns, CalVTP treatments, federal (NFPORS / FACTS) treatments | CAL FIRE, Board of Forestry, NIFC | polygons | statewide |
| `buildings` | Building footprints (about 11.5 million) | Microsoft US Building Footprints | polygons | statewide |

Fire-weather days are the days when the energy release component (ERC, fuel model G) reaches the cell's own 90th
percentile of 2000–2025. Winds are gridMET daily means at 10 m, so gusts are stronger. The daily series are kept
too, for point queries.

Every layer's units, dates, licence and notes are in `backend/geodata/catalog.py` and in `GET /api/geodata/catalog`.
Modelled layers (Wildfire Risk to Communities, canopy height) are models, not measurements. The Wildfire Risk to
Communities values come from the public display service; check them against the source rasters
([RDS-2020-0016-2](https://doi.org/10.2737/RDS-2020-0016-2)) before publishing numbers.

## Using it

**Map.** The *California data (local cache)* panel in the left sidebar lists the layers by group. Checked layers are
drawn as map tiles with a legend; layers that are too detailed for the current zoom say so. Clicking the map adds a
*California data at this point* section to the location summary. `?geo=fuel_model,ignitions` in the URL opens the
map with those layers on.

**HTTP API.**

| Endpoint | Returns |
|---|---|
| `GET /api/geodata/catalog` | Every layer with source, units, dates, licence, legend and whether it is built |
| `GET /api/geodata/tiles/<layer>/<z>/<x>/<y>.png` | Map tile |
| `GET /api/geodata/point?lat=&lng=[&layers=a,b]` | Value of each layer at a point; vectors at or near it; fire-weather wind rose |
| `GET /api/geodata/area?lat=&lng=&radius_m=` or `?bbox=w,s,e,n` | Per layer: mean and percentiles, or class shares, or feature counts and covered area |
| `GET /api/geodata/features/<layer>?lat=&lng=&radius_m=` or `?bbox=` | Vector features with their attributes (`&geometry=true` for GeoJSON) |
| `GET /api/geodata/fire-weather?lat=&lng=` | Fire-weather days per month, wind rose, and the 10 worst days 2000–2025 |

**Python.**

```python
from backend import geodata
geodata.point(-122.43, 37.62)                              # every layer at a point
geodata.area(-122.43, 37.62, radius_m=500)                 # summaries within 500 m
geodata.features("ignitions", lon=-122.43, lat=37.62, radius_m=3000)
w = geodata.raster_window("fuel_model", (-122.47, 37.60, -122.40, 37.65))  # array + transform for computing
```

**AI agent.** The copilot has four tools on top of this: `list_data_layers`, `query_data_layers` (point or radius),
`get_data_features` and `get_fire_weather_history` (`backend/ai/tools.py`).

## Building the cache

```bash
python scripts/data/fetch_gridmet_ca.py                     # daily gridMET 2000-2025 for California (~5.6 GB)
python -m backend.geodata.build gridmet                     # fire-weather layers and daily series
python -m backend.geodata.build vectors                     # FHSZ, fire history, ignitions, treatments, buildings
python -m backend.geodata.build prefetch --all --workers 8  # statewide rasters, then their coarse levels
python -m backend.geodata.build status
```

The statewide rasters are about 2,000 tiles per 30 m layer and 16,500 per 10 m layer. Planetary Computer limits
request rates; tiles that fail are simply fetched again on the next run or when first viewed. The California
boundary used to skip tiles is the Census 2023 cartographic boundary (`ca_boundary_3310.gpkg` in the data root).

## Not included yet

- Parcel attributes (year built, construction): not public statewide.
- Wall-to-wall lidar: 3DEP lidar products cover only part of the state; canopy height comes from the Meta model.
- Hourly historical wind: gridMET is daily. The NOAA HRRR archive on AWS (2014 on) can provide hourly wind for
  specific fires.
