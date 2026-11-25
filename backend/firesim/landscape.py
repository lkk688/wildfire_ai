"""Landscape, buildings, ignition density and fire-weather days for a simulation area, read from the geodata cache."""
from dataclasses import dataclass, field
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from backend import geodata
from backend.geodata import vectors
from backend.geodata.config import DERIVED

from . import fuels
from .spread import Weather

AOIS = {"san_bruno": (-122.52, 37.54, -122.36, 37.70)}
LAYERS = {"fuel": "fuel_model", "slope_deg": "slope", "aspect_deg": "aspect", "cc": "canopy_cover",
          "ch_m": "canopy_height_lf", "cbh_m": "canopy_base_height", "cbd": "canopy_bulk_density"}


@dataclass
class Landscape:
    bbox: tuple
    transform: object
    land: dict
    burnable: np.ndarray
    buildings: object  # GeoDataFrame in EPSG:3310 with row, col, radius_m
    ignition_weight: np.ndarray
    sources: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.burnable.shape

    @property
    def cell(self):
        return self.transform.a


def load(bbox):
    land, tr, src = {}, None, {}
    for k, lid in LAYERS.items():
        w = geodata.raster_window(lid, bbox, res=30)
        a = w["array"][0].astype("float64")
        nd = a == w["nodata"]
        if w["scale"]:
            a = a * w["scale"]
        a[nd] = 0 if k != "aspect_deg" else -1
        if k == "fuel":
            a[nd] = 99
        land[k] = a
        tr = tr or w["transform"]
        src[lid] = f'{w["source"]} ({w["date"]})'
    land["fuel"] = land["fuel"].astype(int)
    burnable = np.isin(land["fuel"], list(fuels.FBFM40))
    h, wd = burnable.shape
    # buildings (Microsoft footprints) -> EPSG:3310, cell of each centroid, equivalent radius
    b = vectors.query("buildings", bbox).to_crs(3310)
    c = b.geometry.centroid
    col = np.floor((c.x - tr.c) / tr.a).astype(int)
    row = np.floor((c.y - tr.f) / tr.e).astype(int)
    keep = (row >= 0) & (row < h) & (col >= 0) & (col < wd)
    b = b[keep].copy()
    b["row"], b["col"] = row[keep].values, col[keep].values
    b["radius_m"] = np.sqrt(b.geometry.area / np.pi)
    b = b.reset_index(drop=True)
    # ignition density: past ignitions (NIFC InFORM) smoothed with a 1 km kernel, plus a floor so no burnable cell is
    # impossible; normalised over burnable cells
    ig = vectors.query("ignitions", bbox).to_crs(3310)
    dens = np.zeros((h, wd))
    if len(ig):
        ic = np.floor((ig.geometry.x - tr.c) / tr.a).astype(int)
        ir = np.floor((ig.geometry.y - tr.f) / tr.e).astype(int)
        ok = (ir >= 0) & (ir < h) & (ic >= 0) & (ic < wd)
        np.add.at(dens, (ir[ok].values, ic[ok].values), 1)
        dens = gaussian_filter(dens, 1000 / tr.a)
    dens = dens + 0.2 * dens[burnable].mean() + 1e-12
    dens[~burnable] = 0
    src["ignitions"] = f"NIFC InFORM ignition points ({len(ig)} in the area)"
    src["buildings"] = f"Microsoft US Building Footprints ({len(b)} buildings)"
    return Landscape(bbox, tr, land, burnable, b, dens / dens.sum(), src)


# ----------------------------------------------------------------------------- fire-weather days (gridMET)
def emc(rh, t_f=80.0):
    """Equilibrium moisture content in % (Simard 1968)."""
    rh = np.asarray(rh, float)
    return np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * t_f,
                    np.where(rh < 50, 2.22749 + 0.160107 * rh - 0.014784 * t_f,
                             21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t_f - 0.483199 * rh))


def fire_weather_days(lon, lat, pct=90):
    """Daily gridMET 2000-2025 at the cell containing (lon, lat), only days with ERC at or above the cell's own
    percentile `pct` (fire-weather days)."""
    dc = DERIVED / "gridmet_daily"
    g = json.loads((dc / "grid.json").read_text())
    la, lo = np.asarray(g["lat"]), np.asarray(g["lon"])
    r, c = int(np.argmin(np.abs(la - lat))), int(np.argmin(np.abs(lo - lon)))
    days = np.load(dc / "days_since_1900.npy")
    v = {}
    for k in g["vars"]:
        x = np.load(dc / f"{k}.npy", mmap_mode="r")[:, r, c].astype("float64")
        x[x == g["nodata"]] = np.nan
        v[k] = x / 10
    df = pd.DataFrame(v)
    df["date"] = [date(1900, 1, 1) + timedelta(days=int(d)) for d in days]
    thr = np.nanpercentile(df["erc"], pct)
    out = df[df["erc"] >= thr].dropna().reset_index(drop=True)
    out.attrs.update(erc_threshold=float(thr), cell=(float(lo[c]), float(la[r])), years=g["years"])
    return out


def weather_of(day):
    """Scenario weather for one historical day. 1-h moisture from the day's minimum humidity at 80 F (gridMET has no
    hourly temperature here); 10-h = 1-h + 1 %; 100-h from gridMET; live fuels cured from June to November."""
    d1 = float(np.clip(emc(day["rmin"]), 2, 30)) / 100
    late = day["date"].month in (6, 7, 8, 9, 10, 11)
    return Weather(wind_ms=float(day["vs"]), wind_from_deg=float(day["th"]), d1=d1, d10=d1 + 0.01,
                   d100=float(day["fm100"]) / 100, herb=0.30 if late else 0.60, woody=0.70 if late else 1.00,
                   label=f'{day["date"]} wind {day["vs"]:.1f} m/s from {day["th"]:.0f} deg, RHmin {day["rmin"]:.0f}%')


def to_lonlat(x, y):
    return Transformer.from_crs(3310, 4326, always_xy=True).transform(x, y)
