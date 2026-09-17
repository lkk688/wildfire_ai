"""
Wildfire Ground Truth Dataset Ingestion & Verification Suite
Automates downloading, filtering, and converting ground truth datasets:
1. WFIGS Interagency Fire Perimeters (National Interagency Fire Center)
2. CAL FIRE Damage Inspection (DINS) Structural Post-Fire Ground Truth
3. NASA FIRMS VIIRS/MODIS Satellite Active Fire Hotspots
4. MTBS (Monitoring Trends in Burn Severity) Historical Burn Perimeters
5. Academic Benchmarks (WildfireSpreadTS / Google NDWS)
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_DATA_DIR = Path("/data/wildfire_benchmark_data")


def ensure_dirs(base_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, Path]:
    """Create subdirectories for each ground truth modality."""
    subdirs = {
        "wfigs": base_dir / "wfigs_perimeters",
        "dins": base_dir / "calfire_dins",
        "firms": base_dir / "nasa_firms",
        "mtbs": base_dir / "mtbs_severity",
        "academic": base_dir / "academic_benchmarks",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


def download_wfigs_perimeters(
    dest_dir: Optional[Path] = None,
    min_acres: float = 500.0,
    limit: int = 100,
) -> Path:
    """
    Download authoritative WFIGS / NIFC wildfire perimeter polygons.
    """
    dest_dir = dest_dir or ensure_dirs()["wfigs"]
    out_file = dest_dir / "wfigs_interagency_perimeters.geojson"
    
    url = (
        "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
        f"?where=poly_GISAcres%3E{min_acres}&outFields=*&f=geojson&resultRecordCount={limit}"
    )
    print(f"[*] Downloading WFIGS fire perimeters (min {min_acres} acres) -> {out_file}...")
    req = urllib.request.Request(url, headers={"User-Agent": "WildfireAI-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_file, "wb") as f:
        f.write(resp.read())
    
    file_size_mb = round(out_file.stat().st_size / (1024 * 1024), 2)
    print(f"[✓] WFIGS downloaded successfully: {file_size_mb} MB")
    return out_file


def download_calfire_dins(
    dest_dir: Optional[Path] = None,
    limit: int = 250,
) -> Path:
    """
    Download CAL FIRE Damage Inspection (DINS) structural point records.
    Contains ground truth for structure damage: Destroyed, Major, Minor, Affected, No Damage.
    """
    dest_dir = dest_dir or ensure_dirs()["dins"]
    out_file = dest_dir / "calfire_dins_structures.geojson"

    url = (
        "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
        "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query"
        f"?where=DAMAGE%20IS%20NOT%20NULL&outFields=*&f=geojson&resultRecordCount={limit}"
    )
    print(f"[*] Downloading CAL FIRE DINS building damage records -> {out_file}...")
    req = urllib.request.Request(url, headers={"User-Agent": "WildfireAI-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_file, "wb") as f:
        f.write(resp.read())

    file_size_kb = round(out_file.stat().st_size / 1024, 2)
    print(f"[✓] CAL FIRE DINS downloaded successfully: {file_size_kb} KB")
    return out_file


def download_nasa_firms(
    dest_dir: Optional[Path] = None,
    horizon: str = "24h",
) -> Path:
    """
    Download NASA FIRMS real-time VIIRS Suomi-NPP active thermal hotspots for USA.
    """
    dest_dir = dest_dir or ensure_dirs()["firms"]
    out_file = dest_dir / f"viirs_snpp_usa_{horizon}.csv"

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/"
        f"SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_{horizon}.csv"
    )
    print(f"[*] Downloading NASA FIRMS VIIRS hotspots ({horizon}) -> {out_file}...")
    req = urllib.request.Request(url, headers={"User-Agent": "WildfireAI-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_file, "wb") as f:
        f.write(resp.read())

    file_size_kb = round(out_file.stat().st_size / 1024, 2)
    print(f"[✓] NASA FIRMS downloaded successfully: {file_size_kb} KB")
    return out_file


def inspect_inventory(base_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Inspect and inventory all downloaded datasets in /data/wildfire_benchmark_data."""
    results = {}
    if not base_dir.exists():
        return {"status": "directory_missing", "base_dir": str(base_dir)}

    for root, dirs, files in os.walk(base_dir):
        rel_root = os.path.relpath(root, base_dir)
        if files:
            file_summaries = []
            for f in files:
                f_path = Path(root) / f
                size_mb = round(f_path.stat().st_size / (1024 * 1024), 3)
                file_summaries.append({"file": f, "size_mb": size_mb})
            results[rel_root] = file_summaries
    return results


if __name__ == "__main__":
    print(f"=== Wildfire Ground Truth Data Acquisition Suite ===")
    dirs = ensure_dirs()
    print(f"Target Directory: {DEFAULT_DATA_DIR}")
    
    # 1. Download / update WFIGS
    try:
        download_wfigs_perimeters(limit=50)
    except Exception as e:
        print(f"[!] WFIGS download failed: {e}")

    # 2. Download / update DINS
    try:
        download_calfire_dins(limit=200)
    except Exception as e:
        print(f"[!] DINS download failed: {e}")

    # 3. Download / update FIRMS
    try:
        download_nasa_firms(horizon="24h")
    except Exception as e:
        print(f"[!] NASA FIRMS download failed: {e}")

    print("\n=== Current Inventory in /data/wildfire_benchmark_data ===")
    inv = inspect_inventory()
    print(json.dumps(inv, indent=2))
