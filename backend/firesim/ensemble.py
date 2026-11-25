"""
Monte Carlo of fires: many (historical fire-weather day, ignition point) pairs on one landscape.

For each fire: arrival time over the landscape for one burning period, then the buildings it reaches:
- direct exposure: building within `expose_m` (default 30 m, about the 100-ft home ignition zone) of a burned cell
  whose flame length reaches `flame_m` (default 1.2 m = 4 ft, the usual limit for direct attack by hand crews), so
  fuel reduction near a house counts when it lowers the flames there, not only when it stops the fire;
- with structure-to-structure spread: exposed buildings ignite neighbours whose footprints are within `sep_m`, one
  hop every `hop_min` minutes, until the burning period ends (sensitivity metric, not a prediction).
Also records, per cell, how many building exposures the fire carried through it (the fire's path from the ignition
to the cell that exposed each building), which is what treatment ranking uses.
"""
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import STRtree

from .spread import Grid, behaviour


@dataclass
class Params:
    minutes: float = 360.0  # one burning period, constant weather, no suppression
    expose_m: float = 30.0
    flame_m: float = 1.2
    sep_m: float = 10.0
    hop_min: float = 30.0


def building_graph(buildings, sep_m):
    geoms = buildings.geometry.values
    tree = STRtree(geoms)
    i, j = tree.query(geoms, predicate="dwithin", distance=sep_m)
    keep = i != j
    return i[keep], j[keep]


def plan(ls, weather_days, n_days, per_day, seed):
    """Fires to simulate: n_days historical fire-weather days drawn at random, per_day ignitions each, ignition cells
    drawn from the ignition density."""
    from .landscape import weather_of
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(weather_days), n_days, replace=len(weather_days) < n_days)
    p = ls.ignition_weight.ravel()
    return [(weather_of(weather_days.iloc[r]), rng.choice(p.size, per_day, p=p)) for r in rows]


_G = {}  # landscape, grid and building graph shared with forked workers


def _init(ls, grid, bgraph, params, credit):
    _G.update(ls=ls, grid=grid, bgraph=bgraph, params=params, credit=credit)


def _one_day(args):
    wx, igns = args
    ls, grid, prm = _G["ls"], _G["grid"], _G["params"]
    fb = behaviour(ls.land, wx)
    g = grid.weights(fb)
    t, pred = grid.arrival(g, igns, prm.minutes)
    h, w = ls.shape
    b = ls.buildings
    bcell = b["row"].values * w + b["col"].values
    burned_n = np.zeros(h * w, np.int32)
    fl_sum = np.zeros(h * w)
    credit = np.zeros(h * w) if _G["credit"] else None
    b_direct = np.zeros(len(b), np.int32)
    b_chain = np.zeros(len(b), np.int32)
    fires = []
    bi, bj = _G["bgraph"]
    for k in range(len(igns)):
        tk = t[k]
        burned = np.isfinite(tk)
        burned_n += burned
        fl_sum[burned] += fb["fl"].ravel()[burned]
        area_ha = burned.sum() * ls.cell ** 2 / 1e4
        if burned.sum() <= 1:
            fires.append(dict(area_ha=area_ha, direct=0, chain=0))
            continue
        flaming = burned & (fb["fl"].ravel() >= prm.flame_m)
        if not flaming.any():
            fires.append(dict(area_ha=area_ha, direct=0, chain=0))
            continue
        dist, (ir, ic) = distance_transform_edt(~flaming.reshape(h, w), sampling=ls.cell, return_indices=True)
        d_b = np.maximum(dist.ravel()[bcell] - ls.cell / 2 - b["radius_m"].values, 0)
        exposed = d_b <= prm.expose_m
        src_cell = (ir.ravel() * w + ic.ravel())[bcell]
        b_direct += exposed
        # structure-to-structure spread within the remaining time
        n_chain = int(exposed.sum())
        if exposed.any() and len(bi):
            nb = len(b)
            t0 = tk[src_cell[exposed]]
            rows = np.concatenate([bi, np.full(exposed.sum(), nb)])
            cols = np.concatenate([bj, np.nonzero(exposed)[0]])
            wts = np.concatenate([np.full(len(bi), prm.hop_min), np.maximum(t0, 1e-3)])
            gb = csr_matrix((wts, (rows, cols)), shape=(nb + 1, nb + 1))
            tb = dijkstra(gb, directed=True, indices=nb, limit=prm.minutes)[:nb]
            reached = np.isfinite(tb) | exposed
            b_chain += reached
            n_chain = int(reached.sum())
        if credit is not None and exposed.any():
            # credit every cell on the fire's path from the ignition to the cell that exposed each building
            wgt = np.bincount(src_cell[exposed], minlength=h * w).astype(float)
            cells = np.nonzero(burned)[0]
            order = cells[np.argsort(-tk[cells])].tolist()
            pr = pred[k]
            acc = wgt  # accumulated down the predecessor tree, latest-burning cells first
            for cidx in order:
                v = acc[cidx]
                if v:
                    p = pr[cidx]
                    if p >= 0:
                        acc[p] += v
            credit += acc
        fires.append(dict(area_ha=area_ha, direct=int(exposed.sum()), chain=n_chain))
    return burned_n, fl_sum, credit, b_direct, b_chain, fires, wx.label


def run(ls, fires_plan, params=None, credit=False, workers=None, grid=None):
    """Simulate every fire in the plan. Returns per-cell burn counts, mean flame length where burned, path credit,
    per-building exposure counts and per-fire results."""
    params = params or Params()
    grid = grid or Grid(ls.burnable, ls.cell)
    bgraph = building_graph(ls.buildings, params.sep_m)
    workers = workers or max(1, min(mp.cpu_count() - 4, len(fires_plan)))
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(ls, grid, bgraph, params, credit)) as pool:
        parts = pool.map(_one_day, fires_plan, chunksize=1)
    h, w = ls.shape
    n = sum(len(ig) for _, ig in fires_plan)
    burned = sum(p[0] for p in parts)
    fl = sum(p[1] for p in parts)
    out = dict(
        n_fires=n,
        burn_probability=(burned / n).reshape(h, w),
        mean_flame_m=np.where(burned > 0, fl / np.maximum(burned, 1), 0).reshape(h, w),
        building_direct=sum(p[3] for p in parts) / n,
        building_chain=sum(p[4] for p in parts) / n,
        fires=[dict(f, weather=p[6]) for p in parts for f in p[5]],
    )
    if credit:
        out["credit"] = (sum(p[2] for p in parts) / n).reshape(h, w)
    return out


def summary(res):
    f = res["fires"]
    return dict(
        n_fires=res["n_fires"],
        mean_area_ha=round(float(np.mean([x["area_ha"] for x in f])), 1),
        buildings_exposed_per_fire=round(float(np.mean([x["direct"] for x in f])), 2),
        buildings_reached_with_house_to_house_per_fire=round(float(np.mean([x["chain"] for x in f])), 2),
        fires_exposing_any_building=round(float(np.mean([x["direct"] > 0 for x in f])), 3),
    )
