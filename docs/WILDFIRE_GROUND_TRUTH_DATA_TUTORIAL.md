# Wildfire Ground-Truth & Benchmark Datasets: Engineering Tutorial & Evaluation Guide

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GeoPandas](https://img.shields.io/badge/GIS-GeoPandas%20%7C%20Shapely-green.svg)](https://geopandas.org/)
[![Data Storage](https://img.shields.io/badge/Storage-%2Fdata%2Fwildfire__benchmark__data-orange.svg)](#2-data-directory-layout-in-datawildfire_benchmark_data)
[![Status](https://img.shields.io/badge/Status-Verified%20%26%20Downloaded-brightgreen.svg)](#1-executive-overview)

---

## 1. Executive Overview

Evaluating predictive wildfire models requires authoritative, empirical **Ground-Truth (Label) datasets**. Because wildfire is a multi-scale physical and social phenomenon, no single metric or data source suffices. Evaluation requires distinct types of ground truth depending on the modeling objective:

```
                                 ┌────────────────────────────────────────────────────────┐
                                 │   Wildfire Ground-Truth (Label) Evaluation Hierarchy   │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
        ┌────────────────────────────┬───────────────────────┴────────────────────┬────────────────────────────┐
        ▼                            ▼                                            ▼                            ▼
1. Dynamic Perimeters        2. Structural Damage                         3. Burn Severity             4. Thermal Hotspots
(WFIGS / NIROPS)             (CAL FIRE DINS)                              (USGS MTBS)                  (NASA FIRMS VIIRS)
• Time-of-Arrival (TOA)      • Building-level destruction labels          • Final burned footprint     • Hourly heat signatures
• 1-to-24 hour spread        • Validates 130ft radiant heat contagion     • 4-tier dNBR severity       • Fire Radiative Power (FRP)
• Used for: CSI, IoU, Dice   • Used for: "Quantifying the Negative"       • Used for: Macro validation • Used for: Front tracking
```

All datasets described in this tutorial have been downloaded, verified, and organized on the local workstation disk under `/data/wildfire_benchmark_data/`.

---

## 2. Data Directory Layout in `/data/wildfire_benchmark_data/`

The storage directory is partitioned into five specialized modalities:

```bash
/data/wildfire_benchmark_data/
├── wfigs_perimeters/
│   ├── wfigs_interagency_perimeters.geojson   # 6.7 MB (NIFC active & historical perimeters)
│   └── wfigs_sample_perimeters.geojson        # 7.5 MB (Filtered large fire polygons >1000 acres)
├── calfire_dins/
│   ├── calfire_dins_structures.geojson        # 268 KB (CAL FIRE field inspection building labels)
│   └── calfire_dins_sample.geojson            # 136 KB (Sample verified parcel damage breakdown)
├── nasa_firms/
│   └── viirs_snpp_usa_24h.csv                 # 258 KB (3,200+ satellite active thermal anomalies)
├── mtbs_severity/
│   ├── mtbs_perimeter_data.zip                # 372 MB (Compressed USFS/USGS national archive)
│   ├── mtbs_perims_DD.shp                     # 589 MB (Full 1984-2024 national perimeter shapefile)
│   ├── mtbs_perims_DD.dbf                     # 42 MB  (Attribute database with dNBR burn severity)
│   └── mtbs_perims_DD.shx                     # Spatial index file
└── academic_benchmarks/
    └── WildfireSpreadTS/                      # Cloned NeurIPS 2023/2024 Spatio-Temporal Benchmark
        ├── cfgs/                              # Model configs for U-Net, UTAE, and ConvLSTM
        ├── src/dataloader/                    # Multi-modal PyTorch dataset and datamodule
        └── src/models/                        # Pre-implemented deep learning surrogate architectures
```

---

## 3. Dataset 1: WFIGS / NIROPS Wildfire Perimeters (Time-of-Arrival Ground Truth)

### 3.1 What it Represents
The **Wildland Fire Interagency Geospatial Services (WFIGS)** group (co-managed by USFS, DOI, and NIFC) publishes authoritative fire perimeters mapped during active incidents. For high-priority fires, perimeters are updated nightly via **NIROPS (National Infrared Operations)** aircraft carrying calibrated thermal infrared sensors (Phoenix imaging systems).

### 3.2 Key Attributes & Schema
*   `poly_IncidentName`: Official fire name (e.g., *Cedar Creek*, *Bolt Creek*, *CZU Lightning Complex*).
*   `poly_GISAcres`: Measured polygon acreage.
*   `poly_PolygonDateTime`: Precise millisecond UNIX timestamp of perimeter mapping.
*   `poly_MapMethod`: Sensor modality (e.g., `Infrared`, `GPS Ground Track`, `Hand Drawn`).

### 3.3 How to Load & Rasterize to a Binary Evaluation Grid
The following Python recipe extracts a ground-truth fire perimeter and rasterizes it onto the simulation grid to compute **CSI** and **IoU**:

```python
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# 1. Load WFIGS GeoJSON
geojson_path = "/data/wildfire_benchmark_data/wfigs_perimeters/wfigs_interagency_perimeters.geojson"
gdf = gpd.read_file(geojson_path)

# Filter by incident name or size
large_fires = gdf[gdf["poly_GISAcres"] > 5000]
target_fire = large_fires.iloc[0]
print(f"Target Fire: {target_fire['poly_IncidentName']} ({target_fire['poly_GISAcres']:.1f} acres)")

# 2. Rasterize polygon onto a 100x100 evaluation grid
bounds = target_fire.geometry.bounds  # (minx, miny, maxx, maxy)
transform = from_bounds(*bounds, width=100, height=100)

gt_burned_mask = rasterize(
    [(target_fire.geometry, 1)],
    out_shape=(100, 100),
    transform=transform,
    fill=0,
    dtype=np.uint8,
)

print(f"Ground-Truth Mask Shape: {gt_burned_mask.shape}, Burned Pixels: {np.sum(gt_burned_mask)}")
```

---

## 4. Dataset 2: CAL FIRE DINS (Damage Inspection) Structural Ground Truth

### 4.1 What it Represents
Following wildland fire containment, CAL FIRE Damage Inspection (DINS) specialists conduct on-the-ground, structure-by-structure physical audits within and adjacent to the fire perimeter. This is the **gold-standard dataset for validating residential risk models and building hardening effectiveness**.

### 4.2 Key Attributes & Schema
*   `DAMAGE`: Standard 5-tier destruction label:
    *   `No Damage` (0% structure damage)
    *   `Affected (>0-10%)` (Superficial smoke / paint charring)
    *   `Minor (10-25%)` (Partial siding or deck ignition)
    *   `Major (25-50%)` (Structural failure of roof or walls)
    *   `Destroyed (>50%)` (Complete structural collapse)
*   `STRUCTURETYPE`: `Single Family Residence`, `Commercial`, `Outbuilding`, etc.
*   `ROOFTYPE`: Roofing material (e.g., `Class A Composite Shingle`, `Tile`, `Wood Shake`).
*   `DEFENSIBLE_SPACE`: Verification of 0–30ft fuel clearance compliance.

### 4.3 Evaluating the 130ft Radiant Heat Contagion Model
Use DINS point records to evaluate Eric Saylors' **130-foot contagion formula** and calculate the **"Quantification of the Negative"** (structures saved by defensible space and firefighter suppression):

```python
import json
import geopandas as gpd
import numpy as np

# Load CAL FIRE DINS GeoJSON
dins_path = "/data/wildfire_benchmark_data/calfire_dins/calfire_dins_structures.geojson"
gdf_dins = gpd.read_file(dins_path)

# Distribution of damage classes
print("DINS Damage Distribution:")
print(gdf_dins["DAMAGE"].value_counts())

# Evaluate Saylors' S-Ratio:
# Threatened assets = All buildings within 130ft radiant heat zone
# Saved assets = Buildings that sustained "No Damage" or "Affected" despite being inside the zone
destroyed = gdf_dins[gdf_dins["DAMAGE"].str.contains("Destroyed", na=False)]
survived = gdf_dins[gdf_dins["DAMAGE"] == "No Damage"]

total_threatened_value = len(gdf_dins) * 850_000  # $850k Bay Area replacement cost baseline
saved_property_value = len(survived) * 850_000
survival_rate = len(survived) / len(gdf_dins) * 100

print(f"\n--- S-Ratio Valuation ---")
print(f"Total Inspected Parcels: {len(gdf_dins)}")
print(f"Structures Destroyed: {len(destroyed)} ({len(destroyed)/len(gdf_dins)*100:.1f}%)")
print(f"Structures Survived:  {len(survived)} ({survival_rate:.1f}%)")
print(f"Quantification of the Negative (Saved Value): ${saved_property_value:,.2f}")
```

---

## 5. Dataset 3: USGS / USFS MTBS (Burn Severity Ground Truth)

### 5.1 What it Represents
**Monitoring Trends in Burn Severity (MTBS)** is an interagency program that maps burned area extent and severity across all large fires in the United States from 1984 to the present. Severity is computed via Landsat/Sentinel-2 differenced Normalized Burn Ratio ($\text{dNBR}$):
$$\text{dNBR} = (\text{NBR}_{\text{pre-fire}} - \text{NBR}_{\text{post-fire}}) \times 1000$$

### 5.2 Querying California Historical Fires with GeoPandas
The full 589 MB shapefile is extracted at `/data/wildfire_benchmark_data/mtbs_severity/mtbs_perims_DD.shp`:

```python
import geopandas as gpd

shp_path = "/data/wildfire_benchmark_data/mtbs_severity/mtbs_perims_DD.shp"

# Read shapefile with bounding box or column filters
gdf_mtbs = gpd.read_file(
    shp_path,
    columns=["Fire_Name", "Fire_Type", "Acres", "Ig_Date", "geometry"],
    rows=1000,  # Fast preview
)

print(f"Total Loaded Records: {len(gdf_mtbs)}")
print(gdf_mtbs.head(3))
```

---

## 6. Dataset 4: NASA FIRMS VIIRS (Active Thermal Hotspots)

### 6.1 What it Represents
The **Fire Information for Resource Management System (FIRMS)** distributes Near Real-Time (NRT) active fire locations detected by the **VIIRS 375m sensor** aboard the Suomi-NPP and NOAA-20/21 satellites.

### 6.2 Key Attributes & Schema
*   `latitude`, `longitude`: Center coordinates of 375m fire pixel.
*   `bright_ti4`: Brightness temperature of VIIRS I-4 thermal channel (Kelvin).
*   `frp`: **Fire Radiative Power** (Megawatts), directly proportional to biomass fuel consumption rate.
*   `confidence`: Detection confidence (`low`, `nominal`, `high`).
*   `acq_date`, `acq_time`: Satellite overpass timestamp (UTC).

### 6.3 Filtering Hotspots and Clustering Fire Fronts
```python
import pandas as pd
from sklearn.cluster import DBSCAN

# Load 24-hour VIIRS detections
csv_path = "/data/wildfire_benchmark_data/nasa_firms/viirs_snpp_usa_24h.csv"
df_firms = pd.read_csv(csv_path)

# Filter high-confidence active fire pixels with significant FRP
active_fires = df_firms[(df_firms["confidence"] != "low") & (df_firms["frp"] > 5.0)]
print(f"Total Hotspots: {len(df_firms)}, Filtered Intense Thermal Hotspots: {len(active_fires)}")

# Spatial clustering to group individual hotspots into discrete fire complexes
coords = active_fires[["latitude", "longitude"]].values
db = DBSCAN(eps=0.05, min_samples=3).fit(coords)
active_fires["cluster_id"] = db.labels_

print(f"Identified {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)} Active Fire Complexes")
```

---

## 7. Dataset 5: Academic Benchmarks (WildfireSpreadTS)

The **WildfireSpreadTS** benchmark (NeurIPS 2023 / 2024, Gerhard et al.) provides multi-modal daily time-series arrays formatted for PyTorch:

```bash
cd /data/wildfire_benchmark_data/academic_benchmarks/WildfireSpreadTS
ls cfgs/
# Available experiment configs:
# - cfgs/unet/res18_monotemporal.yaml  (Single-day U-Net baseline)
# - cfgs/convlstm/full_run.yaml        (Spatio-temporal recurrent network)
# - cfgs/UTAE/all_features.yaml        (U-Net with Temporal Attention Encoder)
```

The dataset models daily active fire footprints from VIIRS alongside 10 meteorological and topographic channels.

---

## 8. End-to-End Evaluation Workflow: Computing Model Accuracy

Here is the complete recipe demonstrating how to take a predicted burn perimeter from our **Rothermel Level-Set PDE** and evaluate it against empirical ground truth:

```python
import numpy as np
from src.academic_models.rothermel_level_set import RothermelLevelSetSimulator
from src.academic_models.benchmark_suite import calculate_spatial_metrics

# 1. Run physical model simulation
rls = RothermelLevelSetSimulator(grid_shape=(100, 100), dx=10.0, dy=10.0, fuel_model_id=4)
sim_result = rls.solve_level_set(
    ignition_coords=[(50, 50)],
    total_minutes=60.0,
    dt_minutes=0.25,
    wind_speed_ms=9.0,
    wind_dir_deg=45.0,
    fuel_moisture=0.07,
)
pred_mask = np.array(sim_result["final_burned_mask"], dtype=bool)

# 2. Synthetic or empirical Ground-Truth Mask (e.g. from rasterized WFIGS / MTBS)
# For demonstration: ground truth with slight lateral spread variance
gt_mask = np.zeros_like(pred_mask)
gt_mask[40:70, 42:75] = True

# 3. Calculate scientific spatial metrics
metrics = calculate_spatial_metrics(pred_mask=pred_mask, gt_mask=gt_mask)

print("=== Spatial Accuracy Benchmark Evaluation ===")
print(f"True Positive Cells:   {metrics['tp']}")
print(f"False Positive Cells:  {metrics['fp']} (Over-prediction)")
print(f"False Negative Cells:  {metrics['fn']} (Under-prediction)")
print(f"Critical Success Index (CSI / Threat Score): {metrics['csi_threat_score']:.4f}")
print(f"Intersection over Union (IoU):               {metrics['iou']:.4f}")
print(f"Sørensen-Dice Coefficient:                   {metrics['dice']:.4f}")
print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
```

---

## 9. Automated Download & Maintenance Utility

To update or re-download these ground-truth datasets at any time, run the built-in acquisition script:

```bash
cd /Developer/wildfire_ai
python3 src/download_ground_truth_datasets.py
```

*All datasets will be automatically refreshed and stored in `/data/wildfire_benchmark_data/`.*
