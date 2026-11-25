"""
Fire-spread scenarios and vegetation-treatment ranking for one area.

    python -m backend.firesim.run --aoi san_bruno [--days 120 --per-day 10] [--budgets 25,50,100,200]

Writes rasters, ranked treatment units and results.json to <GEODATA_ROOT>/derived/firesim/<aoi>/.
"""
import argparse
from dataclasses import replace
import json
import logging
import time

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from backend.geodata.config import DERIVED

from . import ensemble as E
from . import landscape as L
from . import prioritize as P
from .spread import Grid

log = logging.getLogger("firesim")


def _paired(base, treated, key="direct", n_boot=1000, seed=0):
    """Reduction in total buildings exposed (paired by fire), with a bootstrap 90 % interval over fires."""
    b = np.array([f[key] for f in base["fires"]], float)
    t = np.array([f[key] for f in treated["fires"]], float)
    red = 1 - t.sum() / max(b.sum(), 1e-9)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), (n_boot, len(b)))
    boots = 1 - t[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-9)
    return dict(reduction=round(float(red), 4), lo=round(float(np.percentile(boots, 5)), 4),
                hi=round(float(np.percentile(boots, 95)), 4), per_fire_base=round(float(b.mean()), 3),
                per_fire_treated=round(float(t.mean()), 3))


def _tif(path, arr, ls, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:3310", transform=ls.transform, compress="deflate") as d:
        d.write(arr.astype(dtype), 1)


def main(aoi, n_days, per_day, budgets, seed=1, design_fires=400, flame_m=1.2):
    out = DERIVED / "firesim" / (aoi + ("" if flame_m == 1.2 else f"_flame{flame_m:g}") + ("" if seed == 1 else f"_seed{seed}"))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ls = L.load(L.AOIS[aoi])
    cx, cy = (ls.bbox[0] + ls.bbox[2]) / 2, (ls.bbox[1] + ls.bbox[3]) / 2
    days = L.fire_weather_days(cx, cy)
    grid = Grid(ls.burnable, ls.cell)
    prm = E.Params(flame_m=flame_m)
    log.info("landscape %s, %d burnable cells, %d buildings, %d fire-weather days", ls.shape, ls.burnable.sum(),
             len(ls.buildings), len(days))

    planning = E.run(ls, E.plan(ls, days, n_days, per_day, seed), prm, credit=True, grid=grid)
    eval_plan = E.plan(ls, days, n_days, per_day, seed + 1)
    rng = np.random.default_rng(seed + 2)
    p = ls.ignition_weight.ravel()
    design_plan = [(P.DESIGN_OFFSHORE, rng.choice(p.size, 10, p=p)) for _ in range(design_fires // 10)]
    base = {"historical": E.run(ls, eval_plan, prm, grid=grid), "offshore": E.run(ls, design_plan, prm, grid=grid)}
    log.info("baseline done (%.0f s): %s", time.time() - t0, {k: E.summary(v) for k, v in base.items()})

    uid, n_cells = P.units(ls)
    sc = P.scores(ls, uid, n_cells, planning, np.random.default_rng(seed + 3))
    sc.update({f"random_{k}": np.where(np.isfinite(sc["random"]), np.random.default_rng(100 + k).random(n_cells.size),
                                       -np.inf) for k in (1, 2)})
    # iterative simulation: pick units in batches, re-simulating the planning fires on the treated landscape after each
    # batch so the next batch follows where the fires go once the first units are treated
    batch_ha = min(budgets)
    seq, land = [], ls.land
    plan_fires = E.plan(ls, days, n_days, per_day, seed)
    while sum(n_cells[seq]) * ls.cell ** 2 / 1e4 < max(budgets):
        pl = planning if not seq else E.run(replace(ls, land=land), plan_fires, prm, credit=True, grid=grid)
        s_it = P.scores(ls, uid, n_cells, pl, np.random.default_rng(0))["simulation"]
        s_it[seq] = -np.inf
        seq += P.select(s_it, n_cells, batch_ha, ls.cell).tolist()
        land = P.treat(ls.land, P.mask_of(uid, seq))
    log.info("iterative selection done (%.0f s)", time.time() - t0)
    cum = np.cumsum(n_cells[seq]) * ls.cell ** 2 / 1e4
    iter_pick = lambda b: np.array(seq[: int(np.searchsorted(cum, b)) + 1])  # noqa: E731

    strategies = ["simulation_iterative", "simulation", "hybrid", "exposure", "proximity", "hazard", "random", "random_1",
                  "random_2"]
    results = []
    for budget in budgets:
        for s in strategies:
            chosen = iter_pick(budget) if s == "simulation_iterative" else P.select(sc[s], n_cells, budget, ls.cell)
            m = P.mask_of(uid, chosen)
            lt = replace(ls, land=P.treat(ls.land, m))
            row = dict(budget_ha=budget, strategy=s, units=len(chosen), treated_ha=round(m.sum() * ls.cell ** 2 / 1e4, 1))
            for wx_set, plan_ in (("historical", eval_plan), ("offshore", design_plan)):
                r = E.run(lt, plan_, prm, grid=grid)
                row[wx_set] = _paired(base[wx_set], r)
                row[wx_set + "_house_to_house"] = _paired(base[wx_set], r, key="chain")
            results.append(row)
            log.info("%s %s ha: hist %.1f%%  offshore %.1f%%", s, budget, 100 * row["historical"]["reduction"],
                     100 * row["offshore"]["reduction"])

    # rasters
    _tif(out / "burn_probability.tif", planning["burn_probability"], ls)
    _tif(out / "mean_flame_length_m.tif", planning["mean_flame_m"], ls)
    _tif(out / "path_credit.tif", planning["credit"], ls)
    _tif(out / "units.tif", uid, ls, "int32")
    # ranked units (simulation strategy) as polygons, with the other strategies' ranks for comparison
    valid = np.nonzero(n_cells > 0)[0]
    wb = (ls.shape[1] + P.BLOCK - 1) // P.BLOCK
    rank = {s: np.empty(n_cells.size) for s in ("simulation", "exposure", "proximity", "hazard")}
    for s in rank:
        o = np.argsort(-sc[s])
        rank[s][o] = np.arange(1, o.size + 1)
    top = valid[np.argsort(-sc["simulation"][valid])][:1000]
    a, c, f, e = ls.transform.a, ls.transform.c, ls.transform.f, ls.transform.e
    geoms = [box(c + (u % wb) * P.BLOCK * a, f + (u // wb + 1) * P.BLOCK * e, c + (u % wb + 1) * P.BLOCK * a,
                 f + (u // wb) * P.BLOCK * e) for u in top]
    gdf = gpd.GeoDataFrame(dict(unit=top, rank=rank["simulation"][top].astype(int),
                                score=sc["simulation"][top], treatable_ha=n_cells[top] * ls.cell ** 2 / 1e4,
                                rank_exposure=rank["exposure"][top].astype(int),
                                rank_proximity=rank["proximity"][top].astype(int),
                                rank_hazard=rank["hazard"][top].astype(int)), geometry=geoms, crs=3310)
    gdf.to_crs(4326).to_file(out / "treatment_units_ranked.geojson", driver="GeoJSON")
    np.save(out / "building_exposure_historical.npy", base["historical"]["building_direct"])
    summary = dict(
        aoi=aoi, bbox=ls.bbox, cell_m=ls.cell, sources=ls.sources,
        weather=dict(gridmet_cell=days.attrs["cell"], erc_threshold=days.attrs["erc_threshold"],
                     fire_weather_days=len(days), days_sampled=n_days, ignitions_per_day=per_day,
                     offshore_stress_test=P.DESIGN_OFFSHORE.label),
        params=prm.__dict__, candidate_units=int((n_cells > 0).sum()),
        candidate_ha=round(float(n_cells.sum() * ls.cell ** 2 / 1e4), 1),
        planning=E.summary(planning), baseline={k: E.summary(v) for k, v in base.items()},
        iterative_sequence=[int(u) for u in seq], results=results, seconds=round(time.time() - t0))
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=float))
    log.info("wrote %s (%.0f s)", out, time.time() - t0)
    return summary, ls, planning, base, uid, n_cells, sc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="san_bruno")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--per-day", type=int, default=10)
    ap.add_argument("--budgets", default="25,50,100,200")
    ap.add_argument("--flame-m", type=float, default=1.2, help="flame length that counts as exposing a building")
    ap.add_argument("--seed", type=int, default=1, help="planning fires use seed, evaluation fires seed + 1")
    a = ap.parse_args()
    main(a.aoi, a.days, a.per_day, [float(x) for x in a.budgets.split(",")], seed=a.seed, flame_m=a.flame_m)
