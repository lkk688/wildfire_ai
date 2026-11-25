"""One fire from one ignition point, for the API and the agent."""
import math

import numpy as np
from rasterio import features as rfeatures
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

from . import ensemble as E
from . import landscape as L
from .spread import Grid, Weather, behaviour


def default_weather(lon, lat):
    """Fire-weather-day conditions at the point from gridMET 2000-2025: 90th-percentile wind speed from the most
    common direction, 10th-percentile minimum humidity and 100-h moisture."""
    d = L.fire_weather_days(lon, lat)
    sector = ((d["th"] + 22.5) // 45 % 8).astype(int)
    mode = int(sector.mode()[0])
    return dict(wind_ms=float(np.percentile(d["vs"], 90)), wind_from_deg=mode * 45.0,
                rh_min=float(np.percentile(d["rmin"], 10)), fm100=float(np.percentile(d["fm100"], 10)),
                basis=f"gridMET fire-weather days 2000-2025 at this cell (ERC >= {d.attrs['erc_threshold']:.0f}): "
                      f"p90 wind, most common direction, p10 humidity and 100-h moisture")


def simulate(lat, lng, hours=6.0, wind_ms=None, wind_from_deg=None, rh_min=None, fm100=None, radius_km=None,
             expose_m=30.0):
    dw = default_weather(lng, lat)
    given = dict(wind_ms=wind_ms, wind_from_deg=wind_from_deg, rh_min=rh_min, fm100=fm100)
    defaults_used = [k for k, v in given.items() if v is None]
    wind_ms = dw["wind_ms"] if wind_ms is None else float(wind_ms)
    wind_from_deg = dw["wind_from_deg"] if wind_from_deg is None else float(wind_from_deg)
    rh_min = dw["rh_min"] if rh_min is None else float(rh_min)
    fm100 = dw["fm100"] if fm100 is None else float(fm100)
    d1 = float(np.clip(L.emc(rh_min), 2, 30)) / 100
    wx = Weather(wind_ms=wind_ms, wind_from_deg=wind_from_deg, d1=d1, d10=d1 + 0.01, d100=fm100 / 100)
    # window big enough for the fire: generous for wind, capped
    r_km = radius_km or float(np.clip(1.5 + 0.9 * wind_ms * hours / 6, 3, 12))
    dlat = r_km / 111.0
    dlon = r_km / (111.0 * math.cos(math.radians(lat)))
    ls = L.load((lng - dlon, lat - dlat, lng + dlon, lat + dlat))
    t_tr = ls.transform
    x, y = L.Transformer.from_crs(4326, 3310, always_xy=True).transform(lng, lat)
    row, col = int((y - t_tr.f) / t_tr.e), int((x - t_tr.c) / t_tr.a)
    h, w = ls.shape
    if not ls.burnable[row, col]:
        # move to the nearest burnable cell within 300 m
        rr, cc = np.nonzero(ls.burnable)
        if not len(rr):
            return {"success": False, "error": "no burnable fuel near this point"}
        k = int(np.argmin((rr - row) ** 2 + (cc - col) ** 2))
        if math.hypot(rr[k] - row, cc[k] - col) * ls.cell > 300:
            return {"success": False, "error": "no burnable fuel within 300 m (urban, water or bare ground)"}
        row, col = int(rr[k]), int(cc[k])
    grid = Grid(ls.burnable, ls.cell)
    fb = behaviour(ls.land, wx)
    t, _ = grid.arrival(grid.weights(fb), [row * w + col], hours * 60)
    t = t[0].reshape(h, w)
    burned = np.isfinite(t)
    # buildings exposed (within expose_m of the burned area) and house-to-house sensitivity
    prm = E.Params(minutes=hours * 60, expose_m=expose_m)
    E._init(ls, grid, E.building_graph(ls.buildings, prm.sep_m), prm, False)
    res = E._one_day((wx, np.array([row * w + col])))
    fire = res[5][0]
    # isochrones every hour, as lon/lat polygons
    to_ll = L.Transformer.from_crs(3310, 4326, always_xy=True).transform
    iso = []
    for hr in range(1, int(math.ceil(hours)) + 1):
        m = (t <= hr * 60).astype("uint8")
        if not m.any():
            continue
        polys = [shape(g) for g, v in rfeatures.shapes(m, mask=m.astype(bool), transform=ls.transform) if v == 1]
        geom = shp_transform(to_ll, unary_union(polys).simplify(15))
        iso.append({"type": "Feature", "properties": {"hours": hr, "area_ha": round(float(m.sum()) * ls.cell ** 2 / 1e4, 1)},
                    "geometry": mapping(geom)})
    fl = fb["fl"][burned]
    crown = fb["crown"][burned]
    ign_ll = to_ll(t_tr.c + (col + 0.5) * t_tr.a, t_tr.f + (row + 0.5) * t_tr.e)
    return {
        "success": True,
        "ignition": {"lat": round(ign_ll[1], 5), "lng": round(ign_ll[0], 5)},
        "weather": {"wind_ms_10m": round(wind_ms, 1), "wind_from_deg": round(wind_from_deg),
                    "rh_min_pct": round(rh_min), "fuel_moisture_1h_pct": round(d1 * 100, 1), "fm100_pct": round(fm100, 1),
                    "defaults_used_for": defaults_used,
                    "default_basis": dw["basis"]},
        "hours": hours,
        "burned_area_ha": round(float(burned.sum()) * ls.cell ** 2 / 1e4, 1),
        "flame_length_m": {"median": round(float(np.median(fl)), 1) if fl.size else 0,
                           "p90": round(float(np.percentile(fl, 90)), 1) if fl.size else 0},
        "crown_fire_share": {"torching": round(float((crown == 1).mean()), 3) if crown.size else 0,
                             "active": round(float((crown == 2).mean()), 3) if crown.size else 0},
        "buildings_exposed": fire["direct"],
        "buildings_reached_with_house_to_house": fire["chain"],
        "isochrones": {"type": "FeatureCollection", "features": iso},
        "method": "Rothermel surface spread with Scott & Burgan 40 fuel models, Van Wagner / Rothermel crown fire, "
                  "minimum travel time on a 30 m LANDFIRE grid; constant uniform wind; no spotting or suppression. "
                  f"Building exposed = within {expose_m:.0f} m of burned fuel with flames of 1.2 m (4 ft) or more; house-to-house: footprints within "
                  "10 m, one hop per 30 min. A planning scenario, not a forecast.",
        "sources": ls.sources,
    }
