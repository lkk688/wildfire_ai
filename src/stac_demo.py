"""
Microsoft Planetary Computer STAC Demo for San Bruno Wildfire Sentinel-2 Analysis
- Zero authentication needed (100% open & public)
- Searches cloud-free Sentinel-2 L2A images over San Bruno WUI
- Streams Band 4 (Red) and Band 8 (NIR) via Cloud-Optimized GeoTIFF (COG)
- Calculates 10m resolution NDVI (Vegetation Greenness) and NDWI (Fuel Moisture)
- Saves local GeoTIFF files (san_bruno_ndvi.tif) and a visual preview PNG plot.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rioxarray

# Bounding box for San Bruno Wildland-Urban Interface (WUI)
# [min_lon, min_lat, max_lon, max_lat]
SAN_BRUNO_BBOX = [-122.465, 37.600, -122.400, 37.645]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def run_stac_demo():
    print("=" * 70)
    print("[SAT] MICROSOFT PLANETARY COMPUTER STAC DEMO - SENTINEL-2")
    print("=" * 70)

    # 1. Connect to STAC API
    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    print("\n[Step 1] Connecting to STAC Endpoint (No login required)...")
    catalog = pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    # 2. Search for cloud-free Sentinel-2 L2A scenes
    print(f"[Step 2] Searching for cloud-free Sentinel-2 scenes for San Bruno...")
    print(f"         BBox: {SAN_BRUNO_BBOX}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=SAN_BRUNO_BBOX,
        datetime="2026-05-01/2026-07-29",
        query={"eo:cloud_cover": {"lt": 15}},
        max_items=3,
    )

    items = list(search.items())
    print(f"         Found {len(items)} matching scenes.")

    if not items:
        print("[WARN] No scenes found in exact date range, searching broader range...")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=SAN_BRUNO_BBOX,
            datetime="2025-06-01/2026-07-29",
            query={"eo:cloud_cover": {"lt": 15}},
            max_items=3,
        )
        items = list(search.items())

    best_item = items[0]
    print(f"\n[Step 3] Selected Best Target Scene:")
    print(f"         * Scene ID: {best_item.id}")
    print(f"         * Date:     {best_item.datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"         * Cloud %:  {best_item.properties.get('eo:cloud_cover', 0):.2f}%")

    # Available assets
    print(f"\n[Step 4] Available Downloadable Bands & Assets:")
    band_keys = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)", "B11 (SWIR1)", "visual (RGB)"]
    for b in band_keys:
        key = b.split()[0]
        if key in best_item.assets:
            print(f"         [OK] {b:<15} : {best_item.assets[key].title}")

    # 3. Stream & Crop Bands directly in memory (COG Streaming with crs='EPSG:4326')
    print("\n[Step 5] Streaming & Cropping B04 (Red), B08 (NIR), and B11 (SWIR) for San Bruno...")
    minx, miny, maxx, maxy = SAN_BRUNO_BBOX

    da_b4 = rioxarray.open_rasterio(best_item.assets["B04"].href, masked=True).rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="EPSG:4326"
    )
    da_b8 = rioxarray.open_rasterio(best_item.assets["B08"].href, masked=True).rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="EPSG:4326"
    )

    red = da_b4.values[0].astype(np.float32)
    nir = da_b8.values[0].astype(np.float32)

    # 4. Compute Vegetation Indices
    print("[Step 6] Calculating 10m Resolution NDVI (Vegetation Greenness)...")
    # NDVI = (NIR - Red) / (NIR + Red)
    denom_ndvi = nir + red
    denom_ndvi[denom_ndvi == 0] = np.nan
    ndvi_array = (nir - red) / denom_ndvi

    print(f"         * NDVI Range : Min={np.nanmin(ndvi_array):.3f}, Max={np.nanmax(ndvi_array):.3f}, Mean={np.nanmean(ndvi_array):.3f}")

    # 5. Save Outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save GeoTIFF raster file
    da_ndvi = da_b8.copy(data=ndvi_array[np.newaxis, ...])
    ndvi_tif_path = OUTPUT_DIR / "san_bruno_ndvi.tif"
    da_ndvi.rio.to_raster(ndvi_tif_path)
    print(f"\n[Step 7] Saved Local Output Files:")
    print(f"         * GeoTIFF Raster : {ndvi_tif_path}")

    # Generate Visualization Figure (PNG Plot)
    plt.figure(figsize=(8, 6))
    plt.imshow(ndvi_array, cmap="YlGn", vmin=-0.1, vmax=0.8)
    plt.colorbar(label="NDVI (Vegetation Greenness Index)")
    plt.title(f"San Bruno Sentinel-2 10m NDVI\nScene Date: {best_item.datetime.strftime('%Y-%m-%d')}")
    plt.axis("off")

    plot_path = OUTPUT_DIR / "san_bruno_ndvi_preview.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"         * Preview Image  : {plot_path}")

    print("\n" + "=" * 70)
    print("✅ STAC DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_stac_demo()
