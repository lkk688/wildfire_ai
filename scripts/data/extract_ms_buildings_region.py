"""
Extract Microsoft US Building Footprints (California release) for the demo map's region into a parquet file the
backend can query by bounding box.

Source: https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/California.geojson.zip (ODbL), one
feature per line. Output: data/ms_buildings_bayarea.parquet with centroid lon/lat, footprint area (m^2) and the
polygon ring as a JSON string.

  python scripts/data/extract_ms_buildings_region.py [--bbox min_lon,min_lat,max_lon,max_lat]
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

SRC = Path("/data/wildfire_benchmark_data/ms_buildings/California.geojson")
OUT = Path(__file__).resolve().parents[2] / "data" / "ms_buildings_bayarea.parquet"


def ring_stats(ring):
    lon0 = sum(p[0] for p in ring) / len(ring)
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110574.0
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        a += (x1 - lon0) * kx * (y2 - lat0) * ky - (x2 - lon0) * kx * (y1 - lat0) * ky
    return lon0, lat0, abs(a) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default="-122.65,36.85,-121.45,37.95")
    a = ap.parse_args()
    x0, y0, x1, y1 = map(float, a.bbox.split(","))
    rows = []
    with SRC.open() as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith('{"type":"Feature"'):
                continue
            i = line.find("[[[")
            lon, lat = map(float, line[i + 3: line.find("]", i)].split(",")[:2])
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                continue
            ring = json.loads(line)["geometry"]["coordinates"][0]
            clon, clat, area = ring_stats(ring)
            rows.append((clon, clat, round(area, 1), json.dumps([[round(p[0], 6), round(p[1], 6)] for p in ring])))
    df = pd.DataFrame(rows, columns=["lon", "lat", "area_m2", "ring"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(len(df), "buildings ->", OUT)


if __name__ == "__main__":
    main()
