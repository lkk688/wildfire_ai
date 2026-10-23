"""
(Pilot B) Inventory of time-stamped fire perimeters for DINS fires, for evacuation replay.

Source: NIFC Operational Data Archive (ODA) event polygons, FeatureCategory = 'Wildfire Daily Fire Perimeter'.
Each archived row is a *version* of a perimeter; GDB_FROM_DATE is when that version entered the archive,
which is the earliest time the geometry could have been known. We key every perimeter by GDB_FROM_DATE
(never PolygonDateTime, which is stale on many rows) so a replay never sees a geometry before it existed.

Output: output/evac/perimeters/<event>.geojson (all versions, with t_known) and output/evac/perimeter_inventory.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.tabular_baselines_on_pilot import SPLITS  # noqa: E402

OUT = ROOT / "output" / "evac"
BASE = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
SERVICE = {2018: "National_Incident_Feature_Service_2018", 2019: "2019_NIFS_OpenData",
           2022: "Operational_Data_Archive_2022", 2023: "Operational_Data_Archive_2023",
           2024: "Operational_Data_Archive_2024", 2025: "Operational_Data_Archive_2025"}


def polygon_layer(year):
    meta = requests.get(f"{BASE}/{SERVICE[year]}/FeatureServer", params={"f": "json"}, timeout=60).json()
    for lyr in meta.get("layers", []):
        if lyr["name"].lower().replace("_", "").replace(" ", "").startswith("eventpolygon"):
            return f"{BASE}/{SERVICE[year]}/FeatureServer/{lyr['id']}"
    raise RuntimeError(f"no Event_Polygon layer for {year}: {meta.get('layers')}")


def fetch_versions(layer, name, start, days=14):
    t0 = pd.Timestamp(start).tz_localize("America/Los_Angeles").tz_convert("UTC") - pd.Timedelta(hours=12)
    t1 = t0 + pd.Timedelta(days=days)
    where = (f"upper(IncidentName) = '{name.upper()}' AND FeatureCategory = 'Wildfire Daily Fire Perimeter' "
             f"AND GDB_FROM_DATE >= TIMESTAMP '{t0:%Y-%m-%d %H:%M:%S}' AND GDB_FROM_DATE < TIMESTAMP '{t1:%Y-%m-%d %H:%M:%S}'")
    feats, offset = [], 0
    while True:
        r = requests.get(f"{layer}/query", timeout=120, params={
            "where": where, "outFields": "IncidentName,GISAcres,MapMethod,GDB_FROM_DATE,GDB_TO_DATE,IRWINID,FeatureStatus",
            "returnGeometry": "true", "outSR": 4326, "f": "geojson", "resultOffset": offset, "resultRecordCount": 200})
        d = r.json()
        if "error" in d:
            raise RuntimeError(d["error"])
        feats += d.get("features", [])
        if not d.get("properties", {}).get("exceededTransferLimit") and len(d.get("features", [])) < 200:
            break
        offset += 200
    for f in feats:
        f["properties"]["t_known"] = pd.Timestamp(f["properties"]["GDB_FROM_DATE"], unit="ms", tz="UTC").isoformat()
        to = f["properties"].get("GDB_TO_DATE")
        f["properties"]["t_retired"] = (pd.Timestamp(to, unit="ms", tz="UTC").isoformat()
                                        if to and to < 4e12 else None)
    return sorted(feats, key=lambda f: f["properties"]["GDB_FROM_DATE"]), str(t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-records", type=int, default=100)
    ap.add_argument("--years", default="2022,2023,2024,2025")
    a = ap.parse_args()
    years = {int(y) for y in a.years.split(",")}
    s = pd.read_parquet(SPLITS)
    ev = s.groupby("event").agg(n=("GLOBALID", "size"), split=("split", "first"), pos=("y", "sum")).reset_index()
    ev[["name", "start"]] = ev["event"].str.split(" | ", regex=False, expand=True)
    ev["year"] = ev["start"].str[:4].astype(int)
    ev = ev[(ev["n"] >= a.min_records) & ev["year"].isin(years)].sort_values("n", ascending=False)
    (OUT / "perimeters").mkdir(parents=True, exist_ok=True)
    layers, inv = {}, []
    for _, e in ev.iterrows():
        if e["year"] not in layers:
            layers[e["year"]] = polygon_layer(e["year"])
        try:
            feats, t0 = fetch_versions(layers[e["year"]], e["name"], e["start"])
        except Exception as ex:  # noqa: BLE001
            print("FAIL", e["event"], ex, flush=True)
            inv.append({"event": e["event"], "error": str(ex)})
            continue
        acres = [f["properties"]["GISAcres"] for f in feats]
        times = sorted({f["properties"]["t_known"] for f in feats})
        rec = {"event": e["event"], "split": e["split"], "n_dins": int(e["n"]), "n_destroyed": int(e["pos"]),
               "n_versions": len(feats), "n_times": len(times), "first": times[0] if times else None,
               "last": times[-1] if times else None,
               "acres_first": acres[0] if acres else None, "acres_max": max([x for x in acres if x] or [0])}
        inv.append(rec)
        print(rec, flush=True)
        slug = e["event"].replace(" | ", "_").replace(" ", "_")
        (OUT / "perimeters" / f"{slug}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        time.sleep(0.5)
    (OUT / "perimeter_inventory.json").write_text(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
