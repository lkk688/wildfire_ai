"""
Fire-weather summary for the 50-case VLM dev set shown in the web UI.

For each case: ERA5 hourly reanalysis from the Open-Meteo archive API at the
structure location, for the incident start date (CAL FIRE DINS) and the next
2 days, in local time (America/Los_Angeles).  We summarise max 10 m wind and
the direction at that hour, max gust and its direction, min relative humidity,
max temperature.  ERA5 is a ~25-31 km grid and tends to understate local gusts;
it says nothing about whether a Red Flag Warning was in effect.

Output: <benchmark dir>/dev_case_fire_weather.json  {"cases": {case_id: summary}}

Usage: python scripts/data/fetch_dev_case_weather.py [--force]
"""

import argparse
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.service import BENCHMARK_DIR, BENCHMARK_JSON  # noqa: E402

DINS_PARQUET = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
OUT_JSON = BENCHMARK_DIR / "dev_case_fire_weather.json"
API = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "relative_humidity_2m", "temperature_2m"]
SOURCE = "ERA5 reanalysis via Open-Meteo (may understate local gusts)"
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg):
    return None if deg is None else COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def fetch(lat, lon, start: dt.date, end: dt.date, retries=4):
    q = {
        "latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(VARS), "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
        "timezone": "America/Los_Angeles",
    }
    url = API + "?" + urllib.parse.urlencode(q)
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "WildfireAI/1.0"}),
                                        timeout=30) as r:
                return url, json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"{url}: {last}")


def summarise(j):
    h = j["hourly"]
    t = h["time"]

    def argbest(key, fn):
        vals = [(v, i) for i, v in enumerate(h[key]) if v is not None]
        return fn(vals)[1] if vals else None

    iw = argbest("wind_speed_10m", max)
    ig = argbest("wind_gusts_10m", max)
    irh = argbest("relative_humidity_2m", min)
    it = argbest("temperature_2m", max)

    def at(key, i):
        return None if i is None else h[key][i]

    return {
        "max_wind_mph": at("wind_speed_10m", iw),
        "max_wind_time_local": None if iw is None else t[iw],
        "max_wind_dir_deg": at("wind_direction_10m", iw),
        "max_wind_dir_compass": compass(at("wind_direction_10m", iw)),
        "max_gust_mph": at("wind_gusts_10m", ig),
        "max_gust_time_local": None if ig is None else t[ig],
        "max_gust_dir_deg": at("wind_direction_10m", ig),
        "max_gust_dir_compass": compass(at("wind_direction_10m", ig)),
        "min_rh_pct": at("relative_humidity_2m", irh),
        "min_rh_time_local": None if irh is None else t[irh],
        "max_temp_f": at("temperature_2m", it),
        "max_temp_time_local": None if it is None else t[it],
        "n_hours": len(t),
        "grid_latitude": j.get("latitude"),
        "grid_longitude": j.get("longitude"),
        "grid_elevation_m": j.get("elevation"),
        "timezone": j.get("timezone"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    items = json.loads(BENCHMARK_JSON.read_text())["items"]
    dins = pd.read_parquet(DINS_PARQUET, columns=["OBJECTID", "INCIDENTSTARTDATE_ISO"])
    dins = dins.drop_duplicates("OBJECTID").set_index("OBJECTID")
    prev = json.loads(OUT_JSON.read_text()).get("cases", {}) if OUT_JSON.exists() and not args.force else {}
    out = {}
    for it in items:
        cid = str(it["id"])
        if cid in prev and prev[cid].get("available"):
            out[cid] = prev[cid]
            continue
        oid = int(it["id"])
        if oid not in dins.index:
            out[cid] = {"available": False, "reason": "incident start date not found in DINS", "source": SOURCE}
            continue
        start = pd.Timestamp(dins.loc[oid, "INCIDENTSTARTDATE_ISO"]).date()
        end = start + dt.timedelta(days=2)
        rec = {"source": SOURCE, "incident_start": start.isoformat(), "window_start": start.isoformat(),
               "window_end": end.isoformat(), "incident_start_source": "CAL FIRE DINS INCIDENTSTARTDATE"}
        try:
            url, j = fetch(float(it["latitude"]), float(it["longitude"]), start, end)
            rec.update(summarise(j))
            rec["available"] = True
            rec["request_url"] = url
        except Exception as e:  # noqa: BLE001
            rec["available"] = False
            rec["reason"] = f"Open-Meteo request failed: {e}"[:300]
        out[cid] = rec
        print(cid, rec.get("max_wind_mph"), rec.get("max_gust_mph"), rec.get("min_rh_pct"), rec.get("reason", ""),
              flush=True)
        time.sleep(0.5)

    OUT_JSON.write_text(json.dumps({
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE,
        "api": API,
        "notes": "Incident start date + 2 days, hourly, local time. No historical Red Flag Warning data is used.",
        "cases": out,
    }, indent=1))
    print("available:", sum(1 for r in out.values() if r.get("available")), "/", len(out), "->", OUT_JSON)


if __name__ == "__main__":
    main()
