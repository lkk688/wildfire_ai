"""
CAL FIRE DINS Multi-Modal Benchmark Dataset Builder
Constructs a balanced evaluation dataset pairing:
1. High-resolution aerial imagery tiles (ESRI World Imagery, zoom 18)
2. Structural component attributes (Roof, Siding, Vents, Decks, Eaves)
3. Physical fire environment vectors (Wind, Slope, Fuel Moisture)
4. Empirical ground-truth labels (Destroyed vs Survived, Ignition Point, Flame Source)
"""

import os
import math
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple


def latlon_to_tile(lat: float, lon: float, zoom: int = 18) -> Tuple[int, int]:
    """Convert WGS84 lat/lon to Web Mercator XYZ tile numbers."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def download_aerial_tile(lat: float, lon: float, out_path: Path, zoom: int = 18) -> bool:
    """Download aerial orthophoto patch from ESRI World Imagery."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True

    xtile, ytile = latlon_to_tile(lat, lon, zoom)
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    req = urllib.request.Request(url, headers={"User-Agent": "WildfireAI-Benchmark/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            if len(data) > 1000:
                with open(out_path, "wb") as f:
                    f.write(data)
                return True
    except Exception as e:
        print(f"[!] Tile download failed for ({lat}, {lon}): {e}")
    return False


def build_dins_benchmark(
    dins_geojson_path: str = "/data/wildfire_benchmark_data/calfire_dins/calfire_dins_structures.geojson",
    output_dir: str = "/Developer/wildfire_ai/SBFire_SJSU-20260730T030001Z-1-001/benchmark_results",
    target_count: int = 50,
) -> Dict[str, Any]:
    """Build balanced evaluation benchmark from CAL FIRE DINS records."""
    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(dins_geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    destroyed_pool = []
    survived_pool = []

    for feat in data.get("features", []):
        p = feat.get("properties", {})
        dmg = p.get("DAMAGE")
        lat = p.get("LATITUDE")
        lon = p.get("LONGITUDE")

        if not (lat and lon and dmg):
            continue

        item = {
            "id": p.get("OBJECTID"),
            "global_id": p.get("GLOBALID"),
            "incident": p.get("INCIDENTNAME") or p.get("FIRENAME") or "California WUI Fire",
            "county": p.get("COUNTY"),
            "city": p.get("CITY"),
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "structure_type": p.get("STRUCTURETYPE") or "Single Family Residence",
            "year_built": p.get("YEARBUILT"),
            "assessed_value": p.get("ASSESSEDIMPROVEDVALUE"),
            # Structural Vulnerability Components
            "roof_construction": p.get("ROOFCONSTRUCTION") or "Unknown",
            "exterior_siding": p.get("EXTERIORSIDING") or "Unknown",
            "eaves": p.get("EAVES") or "Unknown",
            "vent_screen": p.get("VENTSCREEN") or "Unknown",
            "window_pane": p.get("WINDOWPANE") or "Unknown",
            "deck_on_grade": p.get("DECKPORCHONGRADE") or "None",
            "deck_elevated": p.get("DECKPORCHELEVATED") or "None",
            "fence_attached": p.get("FENCEATTACHEDTOSTRUCTURE") or "None",
            # Ground-Truth Outcome Labels
            "raw_damage": dmg,
            "survival_label": "Destroyed" if ("Destroyed" in dmg or "Major" in dmg) else "Survived",
            "ignition_point": p.get("WHEREFIRESTARTEDONSTRUCTURE") or ("None" if "No Damage" in dmg else "Unknown"),
            "ignition_source": p.get("WHATDIDFIRESTARTFROM") or ("None" if "No Damage" in dmg else "Unknown"),
            "defensive_actions": p.get("DEFENSIVEACTIONS") or "None",
        }

        if item["survival_label"] == "Destroyed":
            destroyed_pool.append(item)
        elif item["raw_damage"] in ["No Damage", "Minor (10-25%)", "Affected (>0-10%)"]:
            survived_pool.append(item)

    half = target_count // 2
    selected_destroyed = destroyed_pool[:half]
    selected_survived = survived_pool[:half]
    benchmark_items = selected_destroyed + selected_survived

    print(f"[*] Assembling Benchmark: {len(selected_destroyed)} Destroyed + {len(selected_survived)} Survived = {len(benchmark_items)} Items")

    # Download aerial imagery tiles for each parcel
    downloaded_count = 0
    for idx, item in enumerate(benchmark_items):
        img_name = f"case_{item['id']}.jpg"
        img_path = img_dir / img_name
        success = download_aerial_tile(item["latitude"], item["longitude"], img_path, zoom=18)
        if success:
            downloaded_count += 1
            item["image_path"] = str(img_path)
            item["image_filename"] = img_name
        else:
            item["image_path"] = None
            item["image_filename"] = None

    print(f"[✓] Downloaded/Verified {downloaded_count}/{len(benchmark_items)} aerial imagery tiles.")

    # Save benchmark catalog
    catalog_path = out_dir / "dins_vlm_benchmark_50.json"
    benchmark_catalog = {
        "dataset_name": "WildfireReason-DINS-50",
        "description": "Balanced Multimodal Benchmark for Wildfire Structural Survival & Causal Vulnerability Reasoning",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases": len(benchmark_items),
        "class_distribution": {
            "Destroyed": len(selected_destroyed),
            "Survived": len(selected_survived),
        },
        "spatial_zoom": 18,
        "imagery_source": "ESRI World Imagery (0.3-0.5m GSD)",
        "items": benchmark_items,
    }

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_catalog, f, indent=2)

    print(f"[✓] Benchmark dataset saved to: {catalog_path}")
    return benchmark_catalog


if __name__ == "__main__":
    build_dins_benchmark()
