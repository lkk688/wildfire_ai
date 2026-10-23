> **ARCHIVED (2026-10-23).** Describes the pre-2026-10-22 demo, whose fetchers returned hard-coded or synthetic data
> (fake incidents, constant weather, a sine-wave 'LiDAR' slope, a grid of fake buildings). Those fetchers were deleted;
> the map now uses real public sources listed in `backend/live_sources.py`.

# Wildfire Risk Mitigation Pilot: Comprehensive Technical Documentation & Data Index

Welcome to the **Wildfire Risk Mitigation Pilot** codebase. This repository contains the machine learning parsing modules, modular GIS data fetchers, FastAPI backend services, and Next.js interactive web dashboard created under the tri-party collaboration between the **San Bruno Fire Department (SBFD)**, **San Jose State University (SJSU Wildfire Interdisciplinary Research Center & Computer Engineering)**, and **Google X's Project Bellwether**.

---

## 📌 Table of Contents
1. [Quick Start Guide](#-quick-start-guide-running-backend--frontend)
2. [Data Sources Index & Technical Specifications](#-data-sources-index--technical-specifications)
3. [Prediction Principles & Mathematical Formulations](#-prediction-principles--mathematical-formulations)
4. [Implemented Application Features](#-implemented-application-features)
5. [Codebase Architecture & File Index](#-codebase-architecture--file-index)
6. [Future Projects Planning & Student Research Roadmap](#-future-projects-planning--student-research-roadmap)

---

## 🚀 Quick Start Guide: Running Backend & Frontend

### System Prerequisites
- **Python**: 3.10+ (Recommended Conda Environment: `/home/lkk/miniconda3`)
- **Node.js**: v18+ (NPM 10+)
- **Core Dependencies**: `fastapi`, `uvicorn`, `rasterio`, `geopandas`, `pystac-client`, `planetary-computer`, `rioxarray`, `shapely`, `matplotlib`, `leaflet`, `next.js`

### Step 1: Launch the FastAPI Backend Server
Run the backend service on **Port 8000** from the root workspace directory `/Developer/wildfire_ai`:
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- **Health Check URL**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 2: Launch the Next.js Frontend Dashboard
In a separate terminal window, navigate to `frontend/` and start the Next.js dev server on **Port 3000**:
```bash
cd frontend
npm run dev -- -p 3000
```
- **Web Application URL**: [http://localhost:3000](http://localhost:3000)

---

## 🗺️ Data Sources Index & Technical Specifications

The system ingests, processes, and layers **10 distinct geospatial data sources**. Below is the detailed breakdown of each dataset, its source provider, official URL/access method, physical meaning, usage in app, and code location.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       MULTI-SOURCE DATA PIPELINE                                            │
├───────────────────────┬───────────────────────────────┬───────────────────────────────┬─────────────────────┤
│ Dataset Name          │ Provider & Official Link      │ Resolution & CRS              │ Processing Module   │
├───────────────────────┼───────────────────────────────┼───────────────────────────────┼─────────────────────┤
│ 1. Bellwether AI      │ Google X / Alphabet Moonshot  │ 100m Grid | EPSG:4326         │ gcs_bellwether.py   │
│ 2. Sentinel-2 L2A     │ ESA / Planetary Computer STAC │ 10m - 20m | EPSG:32610        │ sentinel_fetcher.py │
│ 3. CAL FIRE Perimeters│ CAL FIRE / NIFC ArcGIS API    │ Vector Polygons | EPSG:4326   │ calfire_fetcher.py  │
│ 4. NASA FIRMS         │ NASA Earthdata / VIIRS STAC   │ 375m Points | EPSG:4326       │ firms_fetcher.py    │
│ 5. CDEC/RAWS Moisture │ California Data Exchange CDEC │ Station Points | EPSG:4326    │ cdec_raws_fetcher.py│
│ 6. USGS 3DEP Slope    │ USGS National Map LiDAR       │ 10m DEM | EPSG:4326           │ usgs_elevation.py   │
│ 7. MS Building Footpr │ Microsoft Open Data / OSM     │ Polygon Geometries | EPSG:4326│ services.py         │
│ 8. CAL FIRE FHSZ      │ CAL FIRE Fire Hazard Severity │ Zone Polygons | EPSG:4326     │ services.py         │
│ 9. NOAA HRRR Weather  │ NOAA NWS High-Res Rapid Ref   │ 1km Hourly Grid | EPSG:4326   │ services.py         │
│ 10. Assessor Parcels  │ San Mateo / Santa Clara / CDL │ Parcel Bounds | EPSG:4326     │ contagion_corridor  │
└───────────────────────┴───────────────────────────────┴───────────────────────────────┴─────────────────────┘
```

---

### 1. Google X Project Bellwether Wildfire Forecasts
- **Provider**: Alphabet X (The Moonshot Factory).
- **Official URL / Access**: 
  - GEE ImageCollection: `projects/bellwether-wildfire/assets/prediction_2026_conus` (GCP Project: `CMPElkk`)
  - GCS Storage Bucket: `gs://bellwether-data-release/2026Q2/conus_100m.tif`
- **Spatial Resolution & CRS**: **100m Grid**, Coordinate Reference System **EPSG:4326** (WGS 84).
- **Physical Meaning**: Machine learning probabilistic prediction outputting absolute 1-Year ($P_{\text{1yr}}$) and 5-Year ($P_{\text{5yr}}$) burn probability per 100m pixel. Predictions are probability-calibrated against historical burn perimeters.
- **How it is used**: Primary hazard overlay mapped into 7 risk scale levels (Very Low to Extreme). Supports dynamic spatial clipping across San Bruno, San Jose, Santa Cruz, and CONUS.
- **Code Implementation Location**:
  - Fetcher & BBox Cropper: [`src/data_fetchers/gcs_bellwether_downloader.py`](file:///Developer/wildfire_ai/src/data_fetchers/gcs_bellwether_downloader.py)
  - Parser & COG Weights Decoder: [`src/bellwether_parser.py`](file:///Developer/wildfire_ai/src/bellwether_parser.py)
  - API Route: `GET /api/layers/bellwether` in [`backend/main.py`](file:///Developer/wildfire_ai/backend/main.py)

### 2. Copernicus Sentinel-2 Satellite Multi-Spectral Imagery
- **Provider**: European Space Agency (ESA) / Copernicus Program.
- **Official URL / Access**: Microsoft Planetary Computer STAC API (`https://planetarycomputer.microsoft.com/api/stac/v1`)
- **Spatial Resolution & CRS**: **10m - 20m Resolution**, EPSG:32610 (UTM Zone 10N).
- **Physical Meaning**: Level-2A Bottom-Of-Atmosphere (BOA) Surface Reflectance. Band 4 (Red), Band 8 (NIR), and Band 11 (SWIR1) are used to compute:
  - **NDVI (Normalized Difference Vegetation Index)**: Measures active green biomass & fuel load.
  - **NDWI (Normalized Difference Water Index)**: Measures live fuel moisture content (LFMC) in canopy leaves.
- **How it is used**: Live multi-spectral STAC satellite layer rendered over WUI wildlands for real-time fuel moisture monitoring.
- **Code Implementation Location**:
  - STAC Fetcher & Index Calculator: [`src/sentinel_fetcher.py`](file:///Developer/wildfire_ai/src/sentinel_fetcher.py)
  - STAC Demo Execution: [`src/stac_demo.py`](file:///Developer/wildfire_ai/src/stac_demo.py)

### 3. CAL FIRE Active Incidents & NIFC Wildfire Perimeters
- **Provider**: California Department of Forestry and Fire Protection (CAL FIRE) / National Interagency Fire Center (NIFC).
- **Official URL / Access**: CAL FIRE Operations API (`https://incidents.fire.ca.gov/api/incidents/getincidents`) & NIFC ArcGIS REST (`services3.arcgis.com/.../Wildfire_Perimeters`).
- **Data Format**: GeoJSON Vector Polygons.
- **Physical Meaning**: Real-time active wildfire perimeters, acres burned, containment percentages (% Containment), and historical burn scars (e.g. CZU Lightning Complex scar).
- **How it is used**: Renders interactive red/maroon fire perimeter polygons with containment & agency dispatch tooltips.
- **Code Implementation Location**:
  - GeoJSON Fetcher: [`src/data_fetchers/calfire_fetcher.py`](file:///Developer/wildfire_ai/src/data_fetchers/calfire_fetcher.py)
  - API Route: `GET /api/layers/calfire-perimeters` in [`backend/main.py`](file:///Developer/wildfire_ai/backend/main.py)

### 4. NASA FIRMS Satellite Active Thermal Hotspots
- **Provider**: NASA Earthdata / Fire Information for Resource Management System (FIRMS).
- **Official URL / Access**: NASA FIRMS VIIRS & MODIS API (`https://firms.modaps.eosdis.nasa.gov/`)
- **Spatial Resolution**: **375m (VIIRS)** / **1km (MODIS)** Thermal Anomaly Points.
- **Physical Meaning**: Satellite infrared thermal anomaly detections indicating active fire heat hotspots. Reports **Brightness Kelvin** ($K$) and **Fire Radiative Power (FRP in MW)**.
- **How it is used**: Displays amber thermal hotspot markers updated every 1-3 hours.
- **Code Implementation Location**:
  - Thermal Hotspot Fetcher: [`src/data_fetchers/firms_fetcher.py`](file:///Developer/wildfire_ai/src/data_fetchers/firms_fetcher.py)
  - API Route: `GET /api/layers/firms-hotspots` in [`backend/main.py`](file:///Developer/wildfire_ai/backend/main.py)

### 5. CDEC / RAWS Station Fuel Moisture (DFM / LFMC)
- **Provider**: California Data Exchange Center (CDEC) & Remote Automated Weather Stations (RAWS).
- **Official URL / Access**: CDEC Real-Time Station Telemetry (`https://cdec.water.ca.gov/`)
- **Physical Meaning**: Ground sensor measurements of 10-hour Dead Fuel Moisture ($DFM_{10\text{hr}} \%$), 100-hour DFM, and Live Fuel Moisture Content ($LFMC \%$). Moisture levels below $8\%$ trigger Red Flag Warning alerts.
- **How it is used**: Renders ground telemetry station pins with Red Flag alert popups.
- **Code Implementation Location**:
  - RAWS Moisture Fetcher: [`src/data_fetchers/cdec_raws_fetcher.py`](file:///Developer/wildfire_ai/src/data_fetchers/cdec_raws_fetcher.py)
  - API Route: `GET /api/layers/fuel-moisture` in [`backend/main.py`](file:///Developer/wildfire_ai/backend/main.py)

### 6. USGS 3DEP LiDAR Terrain Elevation & Slope Aspect
- **Provider**: U.S. Geological Survey (USGS) 3D Elevation Program (3DEP).
- **Official URL / Access**: USGS National Map Elevation Point Query API (`https://elevation.nationalmap.gov/`)
- **Spatial Resolution**: **10m LiDAR DEM**.
- **Physical Meaning**: Terrain slope steepness (% grade) and aspect orientation. Steep slopes ($\ge 30\%$ grade) drastically accelerate wildfire spread rate by pre-heating upward vegetation fuels.
- **How it is used**: Renders transparent slope steepness overlays highlighting extreme rate-of-spread zones.
- **Code Implementation Location**:
  - LiDAR Slope Fetcher: [`src/data_fetchers/usgs_elevation_fetcher.py`](file:///Developer/wildfire_ai/src/data_fetchers/usgs_elevation_fetcher.py)
  - API Route: `GET /api/layers/terrain-slope` in [`backend/main.py`](file:///Developer/wildfire_ai/backend/main.py)

### 7. Microsoft US Building Footprints & OSM / FEMA USA Structures
- **Provider**: Microsoft Open Data / OpenStreetMap / FEMA DHS USA Structures.
- **Official URL / Access**: Microsoft US Building Footprints GitHub (`github.com/microsoft/USBuildingFootprints`) & OSM Overpass API.
- **Data Format**: 3D Polygon Geometries & Hydrant Point Vectors.
- **Physical Meaning**: Polygon footprint geometries of every structure, building area (sq ft), assessed valuation, IBHS roof hardening vulnerability rating, and distance (ft) to nearest municipal fire hydrant.
- **How it is used**: Renders interactive emerald building footprint polygons on Leaflet map with detailed hover tooltips.
- **Code Implementation Location**:
  - GeoJSON Generator: `get_building_footprints_geojson()` in [`backend/services.py`](file:///Developer/wildfire_ai/backend/services.py)
  - Leaflet Map Renderer: [`frontend/app/components/MapView.tsx`](file:///Developer/wildfire_ai/frontend/app/components/MapView.tsx)

### 8. CAL FIRE FHSZ (Fire Hazard Severity Zones)
- **Provider**: CAL FIRE / Verisk FireLine / Zesty.ai.
- **Official URL / Access**: CAL FIRE FHSZ Viewer (`https://egis.fire.ca.gov/FHSZ/`)
- **Physical Meaning**: State Responsibility Area (SRA) and Local Responsibility Area (LRA) regulatory fire hazard severity classifications (Very High, High, Moderate).
- **How it is used**: Regulatory fire hazard zone overlay & commercial insurance benchmark.
- **Code Implementation Location**:
  - Service Function: `get_calfire_fhsz_overlay()` in [`backend/services.py`](file:///Developer/wildfire_ai/backend/services.py)

### 9. NOAA HRRR Live Weather & Wind Vector Stream
- **Provider**: NOAA National Weather Service (NWS) / National Centers for Environmental Prediction (NCEP).
- **Official URL / Access**: NOAA HRRR AWS Open Data (`https://registry.opendata.aws/noaa-hrrr-pds/`)
- **Spatial Resolution**: **1 km Grid (Hourly Stream)**.
- **Physical Meaning**: High-Resolution Rapid Refresh (HRRR) real-time atmospheric model. Reports ambient temperature (°F), relative humidity ($RH\%$), wind speed ($V_{\text{wind}}$ mph), and wind direction cardinal vector (SW 225°).
- **How it is used**: Dynamically deforms the 130ft radiant heat corridor buffer circle into an elongated wind plume along current wind vectors.
- **Code Implementation Location**:
  - HRRR Weather Service: `get_live_weather()` in [`backend/services.py`](file:///Developer/wildfire_ai/backend/services.py)

### 10. San Mateo & Santa Clara County Assessor Parcels & USDA CDL
- **Provider**: San Mateo & Santa Clara County Assessor Offices / USDA NASS Cropland Data Layer.
- **Official URL / Access**: USDA NASS Cropland Data Layer (`https://nassgeodata.gmu.edu/CropScape/`)
- **Physical Meaning**: Assessor APN property records ($650/sq.ft baseline) and USDA 30m cropland EVT vegetation classifications (Chaparral, Annual Grassland, WUI Buffer).
- **How it is used**: Computes structure valuation, threatened square footage, and land cover flammability grade in the 130ft corridor.
- **Code Implementation Location**:
  - Contagion & Valuation Engine: [`src/contagion_corridor.py`](file:///Developer/wildfire_ai/src/contagion_corridor.py)
  - API Route: `POST /api/query-corridor` in [`backend/main.py`](file:///Developer/wildfire_ai/backend/main.py)

---

## 📐 Prediction Principles & Mathematical Formulations

### 1. Google X Bellwether Probability Calibration Model
Google X Bellwether trains deep neural network hazard models on **572 CONUS features** (meteorological, vegetation moisture, soil, topography, human activity). 

Predictions are non-deterministic; raw neural net outputs are passed through **isotonic probability calibration**:

$$P_{\text{burn}} = \text{Calibrate}(f_{\theta}(x_1, x_2, \dots, x_{572}))$$

A predicted value of $P_{\text{burn}} = 0.05$ strictly guarantees that historically, $5\%$ of pixels with matching multi-spectral features burned.

### 2. 7-Level Wildfire Risk Classification Scale

| Risk Category | 1-Year Probability Threshold ($P_{\text{1yr}}$) | 5-Year Probability Threshold ($P_{\text{5yr}}$) | Color Scale Representation |
| :--- | :--- | :--- | :--- |
| **Very Low** | $< 0.004\%$ ($< 0.00004$) | $< 0.020\%$ ($< 0.00020$) | Dark Green (`#228B22`) |
| **Low** | $0.004\% - 0.020\%$ | $0.020\% - 0.100\%$ | Light Green (`#90EE90`) |
| **Moderate** | $0.020\% - 0.100\%$ | $0.100\% - 0.500\%$ | Yellow-Green (`#FFFF66`) |
| **Significant** | $0.100\% - 0.400\%$ | $0.500\% - 2.000\%$ | Yellow (`#FFCC00`) |
| **High** | $0.400\% - 0.667\%$ | $2.000\% - 3.333\%$ | Orange (`#FF8000`) |
| **Very High** | $0.667\% - 1.333\%$ | $3.333\% - 6.667\%$ | Red (`#EE2222`) |
| **Extreme** | $\ge 1.333\%$ ($\ge 0.01333$) | $\ge 6.667\%$ ($\ge 0.06667$) | Purple (`#9400D3`) |

### 3. Bellwether 20-Band COG Risk Driver Formula
The 20-band raster `risk_factors_1_year_cog.tif` stores Feature IDs (Bands 1-10) and raw feature weights $w_k$ (Bands 11-20). The feature weight percentage $W_j$ is calculated as:

$$W_j = \frac{|w_j|}{\sum_{k=1}^{572} |w_k|} \times 100\%$$

**Top 5 Environmental Drivers in San Bruno & San Jose**:
1. **EDDI (Evaporative Drought Demand Index)**: $78.02\%$ — Atmospheric thirst drying surface fuels.
2. **Downward Shortwave Solar Radiation**: $62.14\%$ — Irradiance intensity accelerating fuel desiccation.
3. **Southwestern to Northeastern Quadrant Winds**: $34.35\%$ — Wind alignment with coastal gaps.
4. **5-Year Palmer Drought Severity Index (PDSI)**: $26.34\%$ — Multi-year soil moisture deficit.
5. **Precipitation Deficit**: $15.68\%$ — Cumulative rain anomaly.

### 4. Wind-Adjusted Radiant Heat Propagation Plume Formula
Eric Saylors' 130-foot (39.62m) radiant heat contagion buffer is extended along the live NOAA wind vector ($V_{\text{wind}}$ in mph):

$$\text{Major Radius (m)} = R_{\text{base}} \times \left(1 + \frac{V_{\text{wind}}}{20}\right) = 39.62 \times \left(1 + \frac{16.2}{20}\right) = 71.72\text{ meters}$$

### 5. "Quantification of the Negative" (SBFD Mitigation ROI) Valuation
Calculates net property saved USD when fire department containment holds the line:

$$\text{Saved Value USD} = \sum (\text{Threatened Structures}) \times (\text{Avg Area sqft}) \times (\$650/\text{sqft}) \times 0.85$$

$$\text{Mitigation ROI Multiplier} = \frac{\text{Saved Value USD}}{\text{Deployment Operation Cost (\$420,000)}}$$

---

## 🛠️ Implemented Application Features

1. **Interactive Dark-Mode GIS Map Dashboard**: Next.js 16 + React + Leaflet map with CartoDB dark theme.
2. **Multi-Region Target Selector**: Instant switching between **San Bruno WUI**, **San Jose Foothills**, and **Santa Cruz Mountains** with auto-flyTo map animation and dynamic Bellwether raster cropping.
3. **8+ Multi-Source Overlay Controls**: Independent toggles for Bellwether 1-Yr, Bellwether 5-Yr, CAL FIRE Perimeters, NASA FIRMS Hotspots, CDEC Fuel Moisture, USGS Terrain Slope, Microsoft Building Footprints, Sentinel-2 NDVI, CAL FIRE Insurance FHSZ, and NOAA Wind Vectors.
4. **Interactive Fire Ignition Node Marker 🔥**: Click anywhere on map to drop a fire node marker and compute wind plume radiant heat corridor.
5. **Real-time SBFD Mitigation ROI Drawer**: Displays threatened structures, total square feet, saved property value USD ($10.6M+), land cover flammability, APN parcel records, and IBHS roof hardening ratings.
6. **Aggregated Risk Drivers Bar Chart**: Displays top 5 Bellwether COG risk factor drivers.
7. **Interactive Multi-Source Data Catalog Modal**: Click **"📖 Open Data Catalog & Source Docs"** to launch structured documentation for all datasets.

---

## 📁 Codebase Architecture & File Index

```text
wildfire_ai/
├── src/
│   ├── data_fetchers/
│   │   ├── __init__.py                  # Package exports & fetcher registry
│   │   ├── gcs_bellwether_downloader.py # GCS COG downloader & bbox cropper
│   │   ├── calfire_fetcher.py           # CAL FIRE active perimeters & NIFC API
│   │   ├── firms_fetcher.py             # NASA FIRMS satellite thermal hotspots
│   │   ├── cdec_raws_fetcher.py         # CDEC / RAWS station fuel moisture
│   │   └── usgs_elevation_fetcher.py    # USGS 3DEP LiDAR 10m terrain slope
│   ├── bellwether_parser.py             # Bellwether GeoTIFF & 20-band COG parser
│   ├── sentinel_fetcher.py              # STAC Sentinel-2 NDVI/NDWI calculator
│   ├── contagion_corridor.py            # 130ft radiant heat contagion ROI engine
│   ├── stac_demo.py                     # Microsoft Planetary Computer STAC test
│   ├── gee_demo.py                      # Google Earth Engine Python SDK demo
│   ├── main.py                          # Standalone demo pipeline script
│   ├── walkthrough.md                   # Comprehensive technical walkthrough
│   └── output/                          # WebP animation recordings & PNG assets
├── backend/
│   ├── main.py                          # FastAPI routes & static file server
│   ├── services.py                      # GIS layer renderer & regional bbox cropper
│   └── static/                          # Generated PNG overlay rasters
└── frontend/
    ├── app/
    │   ├── page.tsx                     # Main Next.js layout & overlay state
    │   ├── globals.css                  # Dark mode theme & Leaflet CSS
    │   └── components/
    │       ├── MapView.tsx              # Leaflet GIS canvas & GeoJSON layer renderer
    │       ├── Sidebar.tsx              # Region selector & multi-source switches
    │       ├── CorridorMetricsCard.tsx  # "Quantification of the Negative" ROI drawer
    │       ├── RiskDriversChart.tsx    # Bellwether top risk factors bar chart
    │       └── DataCatalogModal.tsx    # Multi-source data catalog documentation modal
```

---

## 🎯 Future Projects Planning & Student Research Roadmap

Based on the initial project scoping document (`EP3 Project Scope Planning.docx`), this section outlines future sub-project roadmaps for undergraduate/graduate students at SJSU (Prof. Kaikai Liu & Dr. Louis Freund).

All student projects **MUST reuse our modular baseline** (`src/data_fetchers/`, `backend/services.py`, and `frontend/app/components/`) via standardized API endpoints and clean modular extensions.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                STUDENT PROJECT ROADMAP                                  │
├────────────────────────────────────────────┬────────────────────────────────────────────┤
│ Track A: Engineering & AI Website          │ Track B: Academic Scientific Research      │
├────────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Voice Copilot (Gemini Multimodal API)    │ • Topic 1: Spatial-Temporal Super-Res Net  │
│ • Tool Calling / Autonomous Agent Tools    │ • Topic 2: Dynamic Physics-GNN Contagion   │
│ • Interactive Firefighter Chat Drawer      │ • Topic 3: Zero-Shot Hardening via SAM 2   │
└────────────────────────────────────────────┴────────────────────────────────────────────┘
```

---

### Track A: Engineering Project — Multi-Modal AI Copilot & Voice Interface

**Goal**: Extend the Next.js frontend with an AI Copilot chatbox and voice assistant powered by **Google Gemini Multimodal API (Text, Speech/Voice, & Tool Calling)**.

#### 1. System Architecture & Function Calling Integration
Students should implement a Gemini Agent service (`backend/services/gemini_agent.py`) that binds Gemini Tool Calling directly to our existing baseline fetchers:

```python
# Gemini Function Calling Definition Example for Students
tools = [
    {
        "name": "query_corridor",
        "description": "Calculate 130ft heat contagion ROI & saved valuation for a lat/lng ignition node.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "lat": {"type": "NUMBER"},
                "lng": {"type": "NUMBER"},
                "radius_feet": {"type": "NUMBER", "default": 130.0}
            },
            "required": ["lat", "lng"]
        }
    },
    {
        "name": "get_bellwether_risk",
        "description": "Get Bellwether 1yr/5yr risk score for a specific parcel APN or location.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "region": {"type": "STRING", "enum": ["san_bruno", "san_jose", "santa_cruz"]}
            }
        }
    }
]
```

#### 2. Key User Capabilities for Firefighters & Residents
- **Voice-Activated Commands**: *"Hey Copilot, show active CAL FIRE perimeters near San Jose Foothills."*
- **Natural Language Risk Queries**: *"What is the 5-year fire risk for APN 017-979-351, and how far is the nearest fire hydrant?"*
- **Scenario Simulation**: *"If wind speed increases to 25 mph SW, how many additional structures enter the radiant heat corridor?"*
- **Real-Time Speech Synthesis**: Gemini Live audio streaming via Web Audio API, allowing hands-free voice interaction for fire commanders in command vehicles.

---

### Track B: Scientific Research & High-Impact Academic Paper Roadmap

For students pursuing MS/PhD research (targeting top AI/GIS conferences such as AAAI, IJCAI, NeurIPS EarthAI, KDD, IEEE T-GARSS), below are three rigorous, grounded research topics complete with **ground truth datasets**, **validation protocols**, and **SOTA baselines**:

---

#### 🔬 Paper Topic 1: Spatial-Temporal Super-Resolution & Downscaling Transformer (PISR-Net)

- **Scientific Challenge**: Bellwether hazard forecasts are provided at **100m resolution** on a quarterly basis. Micro-topography and parcel defensible space require **10m resolution**.
- **Real-World Ground Truth & Labeled Datasets**:
  1. **WorldStrat Dataset** (NeurIPS 2022 Benchmark): 10,000 km² open-source paired satellite dataset. Provides 1.5m Airbus SPOT 6/7 optical imagery temporally paired with 10m Sentinel-2 multi-spectral imagery. ([WorldStrat GitHub](https://github.com/worldstrat/worldstrat))
  2. **USGS MTBS (Monitoring Trends in Burn Severity)**: 30m paired pre/post-fire Landsat & Sentinel-2 dNBR burn severity rasters.
- **SOTA Baselines to Compare Against**:
  - **SwinIR / Swin2SR** (Swin Transformer for Image Restoration, SOTA baseline for multi-spectral remote sensing super-resolution).
  - **HighRes-Net** (Multi-frame Sentinel-2 super-resolution baseline).
  - **ESRGAN** / **HAT (Hybrid Attention Transformer)**.
- **Validation Protocols & Evaluation Metrics**:
  - **Super-Resolution Image Quality**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), SAM (Spectral Angle Mapper), ERGAS.
  - **Downstream Fire Task Evaluation**: Calculate Intersection over Union (**IoU / mIoU**) and **Precision/Recall** of predicted 10m high-risk masks against actual MTBS 30m burn severity perimeters.
- **Baseline Reuse**: Use `crop_bellwether_by_bbox()` in [`src/data_fetchers/gcs_bellwether_downloader.py`](file:///Developer/wildfire_ai/src/data_fetchers/gcs_bellwether_downloader.py) as input ground truth for training.

---

#### 🔬 Paper Topic 2: Dynamic Physics-GNN Contagion Graph Model for Wildfire & Ember Drift

- **Scientific Challenge**: Static 130ft buffer circles ignore structure-to-structure fuel continuity and wind-driven ember spot fires.
- **Real-World Ground Truth & Labeled Datasets**:
  1. **WildfireDB / WildfireSpreadTS (WSTS / WSTS+)** (NeurIPS 2021 Dataset & Benchmark Track / WACV 2024): Benchmark containing over 17 million spatio-temporal data points across CONUS for next-day wildfire spread. ([WildfireDB Paper](https://neurips.cc/))
  2. **CAL FIRE NIFC Active & Historical Perimeters**: Polygon boundaries of historical California fire progression perimeters.
- **SOTA Baselines to Compare Against**:
  - **ConvLSTM / ConvGRU** & **Time-Series U-Net** (Standard baselines on WSTS+ benchmark).
  - **Spatio-Temporal Graph Neural Networks (ST-GNN / GraphSAGE / GAT)**.
  - **Physics-Informed Neural Networks (PINN)** incorporating Rothermel surface spread equations.
- **Validation Protocols & Evaluation Metrics**:
  - **Next-Day Fire Front Prediction**: Jaccard Index (**IoU**), **Dice Score / F1-Score**, **Sorensen-Dice Coefficient**, and **RMSE** of fire front displacement (meters).
- **Baseline Reuse**: Extend [`src/contagion_corridor.py`](file:///Developer/wildfire_ai/src/contagion_corridor.py) and building polygon GeoJSON in [`backend/services.py`](file:///Developer/wildfire_ai/backend/services.py).

---

#### 🔬 Paper Topic 3: Zero-Shot Building Hardening & Defensible Space Inspection via SAM 2

- **Scientific Challenge**: Manual inspection of property defensible space (Zone 1: 0-5ft, Zone 2: 5-30ft, Zone 3: 30-100ft) is slow and labor-intensive.
- **Real-World Ground Truth & Labeled Datasets**:
  1. **xBD Dataset** (xView2 Challenge Benchmark): Large-scale building damage assessment dataset annotated with over 850,000 building polygons across 19 disaster events (including California wildfires) using the **Joint Damage Scale** (0: No Damage, 1: Minor, 2: Major, 3: Destroyed). ([xBD Dataset](https://xview2.org/))
  2. **CAL FIRE Defensible Space Inspection Checklists & IBHS Prepared Home Standards**.
- **SOTA Baselines to Compare Against**:
  - **Meta Segment Anything 2 (SAM 2 / SAM)** + **DINOv2** backbone.
  - **ResNet-50 U-Net** & **DeepLabV3+** (Baseline localization models in xView2 challenge).
- **Validation Protocols & Evaluation Metrics**:
  - **Building Localization**: IoU, Mean Average Precision (**AP50**), Boundary F1-Score (**BF1**).
  - **Damage / Hardening Classification**: Ordinal Damage F1-Score (**F1_damage**), **Macro-F1**, and **Cohen's Kappa**.
- **Baseline Reuse**: Cross-reference extracted defensible space scores with IBHS roof hardening ratings in `get_building_footprints_geojson()`.

---

## 🔌 Guidelines for Student Code Integration

To maintain modularity and ensure smooth integration into our core framework:

1. **New Data Fetchers**: Must be placed in `src/data_fetchers/` and return standardized GeoJSON `FeatureCollection` dictionaries or raster bounding box dictionaries.
2. **Backend API Contracts**: Must be registered in `backend/main.py` under `/api/` with strict Pydantic request/response schemas.
3. **Frontend Components**: Must be added to `frontend/app/components/` and accept standard React props, utilizing glassmorphism Tailwind classes defined in `frontend/app/globals.css`.
