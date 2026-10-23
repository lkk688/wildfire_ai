"""
(Pilot B) Evacuation-decision replay items from archived perimeters; no future information at decision time.

Decision time t: every time a perimeter version enters the NIFC Operational Data Archive (GDB_FROM_DATE) while the
fire is still at most EARLY_FRAC of its final mapped area. Fire state at t = union of the mapped rows live at t
(t_known <= t < t_retired); if none is live yet, the 'Auto-generated for InFORM' placeholder of the reported size
at the ignition point (what was publicly known then). Mapped perimeters lag the fire by hours, so labels use the
eventual outcome rather than a fixed horizon.

Units: community cells (0.005 deg, >= 10 Microsoft building footprints; see ms_building_cells.py) whose building
centroid lies outside the state at t (+EDGE_M) and within MAX_KM of it.
Inputs known at t: distance and bearing to the perimeter, perimeter area and growth over the previous 12 h,
ERA5 surface wind / gust / RH at the fire at hour t (Open-Meteo archive; observed, not forecast).
Labels (future, never shown to the model):
  reached      : cell centroid inside the final mapped perimeter (+100 m tolerance)
  loss         : >= 1 inspected home in the cell destroyed
  reached_24h  : centroid inside the state at t + 24 h (secondary; affected by mapping lag)
Output: output/evac/items.jsonl, output/evac/fire_states/<event>_<t>.geojson
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point, mapping, shape
from shapely.ops import nearest_points, transform, unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.build_research_splits import load_residential  # noqa: E402

EVAC = ROOT / "output" / "evac"
G, MAX_KM, EARLY_FRAC, MIN_GAP_H, EDGE_M = 0.005, 15, 0.5, 1, 100
FWD = Transformer.from_crs(4326, 3310, always_xy=True).transform
INV = Transformer.from_crs(3310, 4326, always_xy=True).transform


def load_versions(event):
    p = EVAC / "perimeters" / f"{event.replace(' | ', '_').replace(' ', '_')}.geojson"
    rows = []
    for f in json.loads(p.read_text())["features"]:
        pr = f["properties"]
        if not f.get("geometry"):
            continue
        g = transform(FWD, shape(f["geometry"]).buffer(0))
        if g.is_empty:
            continue
        rows.append({"geom": g, "auto": "Auto-generated" in str(pr.get("MapMethod")), "t0": pd.Timestamp(pr["t_known"]),
                     "t1": pd.Timestamp(pr["t_retired"]) if pr.get("t_retired") else pd.Timestamp.max.tz_localize("UTC")})
    return rows


def state_at(rows, t):
    live = [r for r in rows if r["t0"] <= t < r["t1"] and r["geom"].area > 1e4]
    mapped = [r["geom"] for r in live if not r["auto"]]
    if mapped:
        return unary_union(mapped)
    auto = [r["geom"] for r in live if r["auto"]]
    return unary_union(auto) if auto else None


def final_state(rows):
    return unary_union([r["geom"] for r in rows if not r["auto"] and r["t1"] == pd.Timestamp.max.tz_localize("UTC")]
                       or [state_at(rows, max(r["t0"] for r in rows if not r["auto"]))])


def wind_series(lat, lon, start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
        "latitude": lat, "longitude": lon, "start_date": str(start.date()), "end_date": str(end.date()),
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,relative_humidity_2m", "timezone": "UTC"}).json()
    h = r["hourly"]
    return pd.DataFrame({"t": pd.to_datetime(h["time"]).tz_localize("UTC"), "ws": h["wind_speed_10m"],
                         "wd": h["wind_direction_10m"], "gust": h["wind_gusts_10m"], "rh": h["relative_humidity_2m"]})


def bearing(a, b):
    return (math.degrees(math.atan2(b.x - a.x, b.y - a.y)) + 360) % 360


def cells_for(dins, event, ms):
    """Communities = 0.005-degree cells with >= 10 Microsoft building footprints near the fire (outcome-independent),
    with DINS inspection counts attached where the cell was inspected (0 destroyed otherwise)."""
    d = dins[dins["event"] == event].copy()
    d["iy"], d["ix"] = (d.lat // G).astype(int), (d.lon // G).astype(int)
    k = d.groupby(["iy", "ix"]).agg(n_dins=("y", "size"), n_destroyed=("y", "sum")).reset_index()
    c = ms[ms["event"] == event].merge(k, on=["iy", "ix"], how="left").fillna({"n_dins": 0, "n_destroyed": 0})
    c = c.rename(columns={"n_buildings": "n"})
    c["pt"] = [Point(FWD(lo, la)) for lo, la in zip(c["lon"], c["lat"])]
    return c


def main():
    inv = [r for r in json.loads((EVAC / "perimeter_inventory.json").read_text()) if r.get("n_times", 0) >= 3]
    dins = load_residential()
    splits = pd.read_parquet(ROOT / "output" / "splits" / "record_splits.parquet")[["GLOBALID", "split"]]
    dins = dins.merge(splits, on="GLOBALID")
    ms = pd.read_parquet(EVAC / "ms_cells.parquet")
    (EVAC / "fire_states").mkdir(parents=True, exist_ok=True)
    out = (EVAC / "items.jsonl").open("w")
    summary = []
    for e in inv:
        rows = load_versions(e["event"])
        if not rows:
            continue
        cells = cells_for(dins, e["event"], ms)
        if cells.empty:
            continue
        if not any(not r["auto"] for r in rows):
            continue
        fin = final_state(rows)
        fin_tol = fin.buffer(EDGE_M)
        times = []
        for t in sorted({r["t0"] for r in rows}):
            st = state_at(rows, t)
            if st is not None and st.area <= EARLY_FRAC * fin.area:
                times.append(t)
        kept = []
        if not times:
            continue
        for t in times:  # drop updates < MIN_GAP_H after the previous decision point
            if not kept or (t - kept[-1]) >= pd.Timedelta(hours=MIN_GAP_H):
                kept.append(t)
        c0 = transform(INV, state_at(rows, kept[0]) or rows[0]["geom"]).centroid
        wx = wind_series(c0.y, c0.x, kept[0] - pd.Timedelta(days=1), kept[-1] + pd.Timedelta(days=1))
        n_items = 0
        for t in kept:
            s = state_at(rows, t)
            if s is None or s.is_empty:
                continue
            s24 = state_at(rows, t + pd.Timedelta(hours=24)) or s
            s_prev = state_at(rows, t - pd.Timedelta(hours=12))
            area_ha = s.area / 1e4
            growth_ha_h = (area_ha - (s_prev.area / 1e4 if s_prev else 0.0)) / 12.0
            w = wx.iloc[(wx["t"] - t.floor("h")).abs().argmin()]
            tag = t.strftime("%Y%m%dT%H%MZ")
            slug = e["event"].replace(" | ", "_").replace(" ", "_")
            mapped = any(not r["auto"] and r["t0"] <= t < r["t1"] for r in rows)
            (EVAC / "fire_states" / f"{slug}_{tag}.geojson").write_text(json.dumps(
                {"type": "Feature", "properties": {"t": t.isoformat(), "area_ha": area_ha, "mapped": mapped},
                 "geometry": mapping(transform(INV, s))}))
            for c in cells.itertuples():
                if s.buffer(EDGE_M).contains(c.pt):
                    continue
                dist = s.distance(c.pt)
                if dist > MAX_KM * 1000:
                    continue
                near = nearest_points(s, c.pt)[0]
                brg = bearing(near, c.pt)  # direction from fire edge to the community
                wind_to = (w["wd"] + 180) % 360  # ERA5 direction is where wind comes FROM
                align = math.cos(math.radians(brg - wind_to))
                rec = {"event": e["event"], "split": e["split"], "t": t.isoformat(), "state_id": f"{slug}_{tag}",
                       "cell": f"{c.iy}_{c.ix}", "lat": c.lat, "lon": c.lon, "n_buildings": int(c.n),
                       "n_dins": int(c.n_dins),
                       "dist_m": round(dist, 1), "bearing_from_fire": round(brg, 1), "downwind_align": round(align, 3),
                       "area_ha": round(area_ha, 1), "growth_ha_h": round(growth_ha_h, 2),
                       "wind_ms": float(w["ws"]) / 3.6, "wind_from_deg": float(w["wd"]), "gust_ms": float(w["gust"]) / 3.6,
                       "rh": float(w["rh"]), "local_hour": int(t.tz_convert("America/Los_Angeles").hour),
                       "state_mapped": mapped, "final_frac": round(s.area / fin.area, 3),
                       "reached": int(fin_tol.contains(c.pt)), "reached_24h": int(s24.contains(c.pt)),
                       "loss": int(c.n_destroyed > 0),
                       "n_destroyed": int(c.n_destroyed)}
                out.write(json.dumps(rec) + "\n")
                n_items += 1
        summary.append({"event": e["event"], "split": e["split"], "decision_points": len(kept), "items": n_items,
                        "final_ha": round(fin.area / 1e4)})
        print(summary[-1], flush=True)
    out.close()
    (EVAC / "items_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
