"""
Where to treat vegetation first, with a fixed area budget, and how much each choice reduces building exposure.

Treatment units are 90 m x 90 m blocks (3 x 3 cells, 0.81 ha) with burnable fuel within `near_m` of a building.
Ranking strategies (all at the same treated area):
- simulation: building exposures the simulated fires carried through the unit (path credit), per hectare;
- proximity: buildings within ~40 m of the unit's vegetation (the defensible-space rule of thumb), per hectare;
- hazard: simulated burn probability x flame length in the unit;
- exposure: proximity weighted by burn probability (vegetation near buildings that fires actually reach);
- hybrid: simulation and exposure scores, each normalised to sum to 1, averaged;
- random.
Rankings come from a planning ensemble; reductions are measured on a separate evaluation ensemble (different
ignitions and weather days), so the simulation strategy is not scored on the fires it was ranked with.
"""
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

from .spread import Weather

BLOCK = 3


def treat(land, mask):
    """Fuel after treatment of the cells in mask: surface fuels reduced to the lightest model of their family,
    canopy thinned from below (base height at least 3 m, bulk density x 0.6)."""
    out = {k: v.copy() for k, v in land.items()}
    f = out["fuel"]
    fam = f // 10 * 10  # 100 grass, 120 grass-shrub, 140 shrub, 160 timber understory, 180 timber litter, 200 slash
    new = np.select([(fam == 100), (fam == 120), (fam == 140), (fam == 160), (fam == 180), (fam == 200)],
                    [101, 121, 121, 161, 181, 183], f)
    f[mask] = new[mask]
    forest = mask & (out["cbd"] > 0)
    out["cbh_m"][forest] = np.maximum(out["cbh_m"][forest], 3.0)
    out["cbd"][forest] *= 0.6
    return out


def units(ls, near_m=500.0):
    """Unit id per cell (-1 where not treatable) and per-unit treatable cell counts."""
    h, w = ls.shape
    bmask = np.zeros((h, w), bool)
    bmask[ls.buildings["row"].values, ls.buildings["col"].values] = True
    dist = distance_transform_edt(~bmask, sampling=ls.cell)
    treatable = ls.burnable & (dist <= near_m)
    rr, cc = np.indices((h, w))
    uid = (rr // BLOCK) * ((w + BLOCK - 1) // BLOCK) + cc // BLOCK
    uid = np.where(treatable, uid, -1)
    n = np.bincount(uid[uid >= 0], minlength=uid.max() + 1)
    small = n < 3  # units with fewer than 3 treatable cells are dropped
    uid = np.where((uid >= 0) & ~small[np.maximum(uid, 0)], uid, -1)
    return uid, np.bincount(uid[uid >= 0], minlength=n.size)


def scores(ls, uid, n_cells, planning, rng):
    """Per-unit score for every strategy (higher = treat first)."""
    h, w = ls.shape
    valid = uid >= 0
    per_unit = lambda a: np.bincount(uid[valid], weights=a[valid], minlength=n_cells.size)  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = per_unit(planning["credit"]) / n_cells
        bcount = np.zeros((h, w))
        np.add.at(bcount, (ls.buildings["row"].values, ls.buildings["col"].values), 1)
        r = int(np.ceil(40 / ls.cell))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        near = convolve(bcount, ((yy ** 2 + xx ** 2) * ls.cell ** 2 <= (40 + ls.cell / 2) ** 2).astype(float),
                        mode="constant")
        prox = per_unit(near) / n_cells
        hazard = per_unit(planning["burn_probability"] * planning["mean_flame_m"]) / n_cells
        expo = per_unit(planning["burn_probability"] * near) / n_cells
    rand = rng.random(n_cells.size)
    s = dict(simulation=sim, proximity=prox, hazard=hazard, exposure=expo, random=rand)
    for k in s:
        s[k] = np.where(n_cells > 0, np.nan_to_num(s[k], nan=0.0), -np.inf)
    norm = lambda a: np.where(np.isfinite(a), a, 0) / max(np.where(np.isfinite(a), a, 0).sum(), 1e-12)  # noqa: E731
    s["hybrid"] = np.where(n_cells > 0, 0.5 * norm(s["simulation"]) + 0.5 * norm(s["exposure"]), -np.inf)
    return s


def select(score, n_cells, budget_ha, cell_m):
    """Highest-scoring units until the treated area reaches the budget."""
    order = np.argsort(-score)
    order = order[np.isfinite(score[order])]
    area = np.cumsum(n_cells[order]) * cell_m ** 2 / 1e4
    return order[: int(np.searchsorted(area, budget_ha)) + 1]


def mask_of(uid, chosen):
    return np.isin(uid, chosen) & (uid >= 0)


DESIGN_OFFSHORE = Weather(wind_ms=9.0, wind_from_deg=45.0, d1=0.04, d10=0.05, d100=0.08, herb=0.30, woody=0.70,
                          label="stress test: dry offshore (NE) wind 9 m/s, 1-h 4 %, 100-h 8 %, cured")
