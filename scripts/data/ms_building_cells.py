"""
(Pilot B) Outcome-independent community inventory from Microsoft US Building Footprints (California, release 1,
imagery mostly 2018-2020, i.e. before every replay fire).

DINS only inspects structures in and near the burn, so it has almost no communities the fire never reached.
Here every 0.005-degree cell with >= MIN_BUILDINGS footprints within REGION_KM of a fire's final mapped perimeter
is a candidate community, whatever happened to it.

Output: output/evac/ms_cells.parquet  (event, iy, ix, n_buildings, lat, lon)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.build_evac_replay import FWD, INV, final_state, load_versions  # noqa: E402

SRC = Path("/data/wildfire_benchmark_data/ms_buildings/California.geojson")
EVAC = ROOT / "output" / "evac"
G, MIN_BUILDINGS, REGION_KM = 0.005, 10, 16
NUM = re.compile(r"\[\[\[(-?[\d.]+),(-?[\d.]+)\]")


def main():
    inv = [r for r in json.loads((EVAC / "perimeter_inventory.json").read_text()) if r.get("n_times", 0) >= 3]
    regions = []
    for e in inv:
        rows = load_versions(e["event"])
        if not any(not r["auto"] for r in rows):
            continue
        reg = transform(INV, final_state(rows).buffer(REGION_KM * 1000))
        regions.append((e["event"], reg.bounds, reg))
    print(len(regions), "fire regions", flush=True)
    counts = {}
    with SRC.open() as f:
        for i, line in enumerate(f):
            m = NUM.search(line)
            if not m:
                continue
            lon, lat = float(m.group(1)), float(m.group(2))
            for ev, (x0, y0, x1, y1), _ in regions:
                if x0 <= lon <= x1 and y0 <= lat <= y1:
                    k = (ev, int(lat // G), int(lon // G))
                    c = counts.setdefault(k, [0, 0.0, 0.0])
                    c[0] += 1
                    c[1] += lat
                    c[2] += lon
            if i % 2_000_000 == 0:
                print(i, len(counts), flush=True)
    from shapely.geometry import Point
    rows = []
    reg_of = {ev: reg for ev, _, reg in regions}
    for (ev, iy, ix), (n, sla, slo) in counts.items():
        if n < MIN_BUILDINGS:
            continue
        lat, lon = sla / n, slo / n
        if reg_of[ev].contains(Point(lon, lat)):
            rows.append({"event": ev, "iy": iy, "ix": ix, "n_buildings": n, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows)
    df.to_parquet(EVAC / "ms_cells.parquet")
    print(df.groupby("event").size().to_string())


if __name__ == "__main__":
    main()
