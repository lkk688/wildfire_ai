"""
(Pilot A) Objective wildland-interface features per community cell from ESA WorldCover 10 m (2021, v200).

WorldCover 2021 is pre-fire only for fires from 2022 on; for earlier fires it can show the burn scar, so features are
marked valid=False there and must not be used for those fires.

Per cell (centre of the 0.005-degree cell; homes centroid for distance):
  veg_frac_cell   : share of tree/shrub/grass (classes 10/20/30) inside the cell
  veg_frac_1km    : same within a 1 km radius
  built_frac_1km  : share of built-up (50) within 1 km
  dist_wildland_m : distance from the homes' centroid to the nearest pixel of a contiguous wildland patch >= 50 ha
Output: output/community/worldcover.jsonl
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "community"
URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
G, HALF_KM = 0.005, 3.0
WILD = (10, 20, 30)


def tile_name(lat, lon):
    la, lo = int(math.floor(lat / 3) * 3), int(math.floor(lon / 3) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def window(lat, lon):
    dlat = HALF_KM / 111.0
    dlon = HALF_KM / (111.0 * math.cos(math.radians(lat)))
    with rasterio.open(URL.format(tile=tile_name(lat, lon))) as src:
        w = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform)
        a = src.read(1, window=w, boundless=True, fill_value=0)
        tr = src.window_transform(w)
    return a, tr


def feats(rec):
    lat_c, lon_c = rec["lat_c"], rec["lon_c"]
    a, tr = window(lat_c, lon_c)
    res_y = abs(tr.e) * 111000.0
    res_x = abs(tr.a) * 111000.0 * math.cos(math.radians(lat_c))
    rows, cols = a.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    lat_g = tr.f + (yy + 0.5) * tr.e
    lon_g = tr.c + (xx + 0.5) * tr.a
    wild = np.isin(a, WILD)
    in_cell = (np.floor(lat_g / G) == rec["iy"]) & (np.floor(lon_g / G) == rec["ix"])
    dy = (lat_g - lat_c) * 111000.0
    dx = (lon_g - lon_c) * 111000.0 * math.cos(math.radians(lat_c))
    r1 = dx ** 2 + dy ** 2 <= 1000 ** 2
    lab, n = ndimage.label(wild)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * res_x * res_y / 1e4
    big = np.isin(lab, np.where(sizes >= 50)[0] + 1)
    dist = ndimage.distance_transform_edt(~big, sampling=(res_y, res_x))
    hy = int(round((rec["lat_mean"] - tr.f) / tr.e))
    hx = int(round((rec["lon_mean"] - tr.c) / tr.a))
    d = float(dist[min(max(hy, 0), rows - 1), min(max(hx, 0), cols - 1)]) if big.any() else float("nan")
    return {"veg_frac_cell": float(wild[in_cell].mean()) if in_cell.any() else float("nan"),
            "veg_frac_1km": float(wild[r1].mean()), "built_frac_1km": float((a[r1] == 50).mean()),
            "dist_wildland_m": d}


def main():
    cells = [json.loads(x) for x in (OUT / "cells.jsonl").read_text().splitlines()]
    out = OUT / "worldcover.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["cell_id"] for x in out.read_text().splitlines()}
    with out.open("a") as f:
        for i, r in enumerate(cells):
            if r["cell_id"] in done:
                continue
            try:
                ft = feats(r)
            except Exception as ex:  # noqa: BLE001
                ft = {"error": str(ex)[:200]}
            ft.update({"cell_id": r["cell_id"], "valid": int(r["fire_start"][:4]) >= 2022})
            f.write(json.dumps(ft) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, len(cells), r["cell_id"], ft, flush=True)


if __name__ == "__main__":
    main()
