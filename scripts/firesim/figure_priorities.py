"""
Figure of a firesim run: burn probability, building exposure, path credit with the treatment units each strategy picks,
and exposure reduction against treated area.

    python scripts/firesim/figure_priorities.py --aoi san_bruno --out <png>
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.firesim import landscape as L, prioritize as P  # noqa: E402
from backend.geodata.config import DERIVED  # noqa: E402

COL = {"simulation_iterative": "#2a78d6", "hybrid": "#eb6834", "exposure": "#1baf7a", "proximity": "#eda100",
       "hazard": "#e87ba4", "random": "#8a8984"}
LABEL = {"simulation_iterative": "Simulation, re-run after each 25 ha",
         "hybrid": "Hybrid (fire paths + exposure)", "proximity": "Proximity (buildings within 40 m)",
         "exposure": "Exposure (burn prob. x buildings within 40 m)",
         "hazard": "Hazard (burn prob. x flame length)", "random": "Random (mean of 3)"}
INK, MUTED = "#0b0b0b", "#52514e"


def main(aoi, out, budget_map=100.0, sub=None):
    d = DERIVED / "firesim" / (sub or aoi)
    res = json.loads((d / "results.json").read_text())
    rd = lambda n: rasterio.open(d / n).read(1)  # noqa: E731
    bp, fl, credit = rd("burn_probability.tif"), rd("mean_flame_length_m.tif"), rd("path_credit.tif")
    ls = L.load(L.AOIS[aoi])
    uid, n_cells = P.units(ls)
    sc = P.scores(ls, uid, n_cells, dict(credit=credit, burn_probability=bp, mean_flame_m=fl), np.random.default_rng(4))
    exp = np.load(d / "building_exposure_historical.npy")
    h, w = ls.shape
    ext = (0, w * ls.cell / 1000, 0, h * ls.cell / 1000)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 10.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.82], hspace=0.12, wspace=0.18)
    base = np.where(ls.burnable, 0.93, np.where(ls.land["fuel"] == 98, 0.80, 0.97))  # water darker, urban/bare light

    def panel(ax, title):
        ax.imshow(base, cmap="gray", vmin=0, vmax=1, extent=ext)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a  Burn probability, 6-h fires on fire-weather days")
    m = ax.imshow(np.ma.masked_less_equal(bp, 0), cmap="Oranges", extent=ext, vmin=0, vmax=np.percentile(bp[bp > 0], 99))
    fig.colorbar(m, ax=ax, fraction=0.035, pad=0.01).set_label("share of simulated fires")
    ax.plot([0.6, 2.6], [0.6, 0.6], color=INK, lw=2); ax.text(1.6, 0.85, "2 km", ha="center", color=INK)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b  Buildings exposed (flames >= 1.2 m within 30 m)")
    b = ls.buildings
    hit = exp > 0
    xs, ys = (b["col"].values + 0.5) * ls.cell / 1000, (h - b["row"].values - 0.5) * ls.cell / 1000
    ax.scatter(xs[~hit], ys[~hit], s=0.2, color="#c3c2b7", lw=0)
    sca = ax.scatter(xs[hit], ys[hit], s=3, c=exp[hit], cmap="Reds", lw=0, vmin=0, vmax=np.percentile(exp[hit], 98))
    fig.colorbar(sca, ax=ax, fraction=0.035, pad=0.01).set_label("share of fires exposing the building")

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, f"c  Fire paths to buildings, and the first {budget_map:.0f} ha each strategy treats")
    cm = np.ma.masked_less_equal(credit, 0)
    ax.imshow(np.log10(cm), cmap="Blues", extent=ext, alpha=0.9)
    seq = np.array(res["iterative_sequence"])
    cum = np.cumsum(n_cells[seq]) * ls.cell ** 2 / 1e4
    for s in ("simulation_iterative", "hybrid", "exposure", "proximity"):
        ch = seq[: int(np.searchsorted(cum, budget_map)) + 1] if s == "simulation_iterative" else \
            P.select(sc[s], n_cells, budget_map, ls.cell)
        mk = P.mask_of(uid, ch)
        ax.contour(mk.astype(float), levels=[0.5], colors=[COL[s]], linewidths=1.1, extent=ext,
                   origin="upper")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COL[s], lw=2, label=LABEL[s]) for s in ("simulation_iterative", "hybrid", "exposure", "proximity")],
              loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    for k, (wx, title) in enumerate((("historical", "d  Historical fire-weather days (held-out fires)"),
                                     ("offshore", "e  Stress test: dry offshore wind (units ranked on historical days)"))):
        ax = fig.add_subplot(gs[1, 1:3] if k else gs[1, 0])
        rows = res["results"]
        budgets = sorted({r["budget_ha"] for r in rows})
        placed = []
        for s in ("simulation_iterative", "hybrid", "exposure", "proximity", "hazard", "random"):
            if s == "random":
                vals = [[r[wx]["reduction"] for r in rows if r["budget_ha"] == bgt and r["strategy"].startswith("random")]
                        for bgt in budgets]
                y = np.array([np.mean(v) for v in vals]) * 100
                lo = hi = None
            else:
                rr = [next(r for r in rows if r["budget_ha"] == bgt and r["strategy"] == s) for bgt in budgets]
                y = np.array([r[wx]["reduction"] for r in rr]) * 100
                lo = np.array([r[wx]["lo"] for r in rr]) * 100
                hi = np.array([r[wx]["hi"] for r in rr]) * 100
            ax.plot(budgets, y, color=COL[s], lw=2, marker="o", ms=4, ls="--" if s == "random" else "-",
                    label=LABEL[s])
            if lo is not None:
                ax.fill_between(budgets, lo, hi, color=COL[s], alpha=0.12, lw=0)
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            if all(abs(y[-1] - p_) > 0.035 * max(span, 1) for p_ in placed):  # skip labels that would collide
                ax.text(budgets[-1] * 1.03, y[-1], f"{y[-1]:.1f}%", va="center", fontsize=8, color=INK)
                placed.append(y[-1])
        ax.set_xscale("log")
        ax.set_xticks(budgets); ax.set_xticklabels([f"{x:g}" for x in budgets])
        ax.set_xlim(budgets[0] * 0.85, budgets[-1] * 1.35)
        ax.set_xlabel("hectares treated")
        ax.set_ylabel("reduction in buildings exposed (%)")
        ax.grid(axis="y", color="#e7e6e2", lw=0.8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        base_n = res["baseline"][wx]["buildings_exposed_per_fire"]
        ax.text(0.01, 0.97, f"untreated: {base_n:.1f} buildings exposed per fire", transform=ax.transAxes, va="top",
                fontsize=8, color=MUTED)
        if k == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8, frameon=False)
    fig.suptitle(f"Fire-spread scenarios and vegetation-treatment priorities, {aoi.replace('_', ' ').title()} "
                 f"({res['planning']['n_fires']} planning fires, {res['baseline']['historical']['n_fires']} evaluation fires)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="san_bruno")
    ap.add_argument("--sub", default=None, help="results folder name if not the aoi (e.g. san_bruno_flame0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.aoi, a.out, sub=a.sub)
