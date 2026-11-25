"""
Download daily gridMET fire-weather variables for California (4 km, 1/24 degree) from the University of Idaho
THREDDS NetCDF Subset Service, one file per variable and year.

    python scripts/data/fetch_gridmet_ca.py [--years 2000-2025] [--workers 4]

Output: /data/wildfire_benchmark_data/ca/raw/gridmet/<var>_<year>.nc (netCDF3, int16 with scale_factor).
backend/geodata/build.py turns these into the fire-weather scenario layers.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

OUT = Path("/data/wildfire_benchmark_data/ca/raw/gridmet")
NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{v}_1979_CurrentYear_CONUS.nc"
VARS = {  # short name -> grid name in the aggregated dataset
    "vs": "daily_mean_wind_speed",               # m/s, 10 m, daily mean
    "th": "daily_mean_wind_direction",           # degrees clockwise from north (direction the wind comes from)
    "erc": "daily_mean_energy_release_component-g",  # NFDRS ERC, fuel model G
    "rmin": "daily_minimum_relative_humidity",   # %
    "fm100": "dead_fuel_moisture_100hr",          # %
}
CA = dict(north=42.1, south=32.4, west=-124.6, east=-114.0)


def one(v, year):
    p = OUT / f"{v}_{year}.nc"
    if p.exists() and p.stat().st_size > 1_000_000:
        return f"skip {p.name}"
    params = dict(var=VARS[v], **CA, time_start=f"{year}-01-01T00:00:00Z", time_end=f"{year}-12-31T00:00:00Z",
                  accept="netcdf")
    for attempt in range(4):
        try:
            r = requests.get(NCSS.format(v=v), params=params, timeout=600)
            r.raise_for_status()
            tmp = p.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(p)
            return f"ok {p.name} {len(r.content) / 1e6:.0f} MB"
        except Exception as e:  # noqa: BLE001
            time.sleep(10 * (attempt + 1))
            err = e
    return f"FAIL {p.name}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2000-2025")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(v, y) for y in range(y0, y1 + 1) for v in VARS]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda j: one(*j), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
