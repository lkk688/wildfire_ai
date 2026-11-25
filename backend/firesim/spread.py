"""
Fire behaviour on a raster and minimum-travel-time spread (Finney 2002 idea, solved as a shortest-path problem).

Each burnable cell gets a head-fire spread rate, its direction (wind and upslope combined as vectors) and an elliptical
spread-rate shape. Travel time along an edge between neighbour cells is distance / spread rate in that direction,
averaged over the two cells; Dijkstra from the ignition gives the arrival time everywhere. 16 neighbours keep the
fire shape close to the ellipse. No spotting; wind is uniform in speed and direction over the landscape.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import fuels

FT = 0.3048
MPH = 0.44704
OFFSETS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
           (1, 2), (2, 1), (2, -1), (1, -2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]


@dataclass
class Weather:
    wind_ms: float  # 10-m wind speed, m/s (gridMET "vs")
    wind_from_deg: float  # direction the wind blows from, degrees clockwise from north
    d1: float  # dead fuel moisture, fractions
    d10: float
    d100: float
    herb: float = 0.30
    woody: float = 0.70
    foliar: float = 1.00  # foliar moisture (fraction) for crown-fire initiation
    label: str = ""


def behaviour(land, wx):
    """Per-cell fire behaviour for one weather scenario.
    land: dict of arrays fuel (FBFM40 codes), slope_deg, aspect_deg, cc (%), ch_m, cbh_m, cbd (kg/m3).
    Returns dict of arrays: ros (m/min, head), dir (radians, azimuth of max spread), ecc, fl (m), crown (0/1/2)."""
    fuel = land["fuel"]
    shape = fuel.shape
    ros = np.zeros(shape); fl = np.zeros(shape); ecc = np.zeros(shape); crown = np.zeros(shape, np.int8)
    direc = np.zeros(shape)
    m = dict(d1=wx.d1, d10=wx.d10, d100=wx.d100, herb=wx.herb, woody=wx.woody)
    u20_ftmin = wx.wind_ms / 1.15 / FT * 60  # 10 m -> 20 ft wind, ft/min
    to_az = np.deg2rad((wx.wind_from_deg + 180) % 360)
    slope = np.tan(np.deg2rad(land["slope_deg"]))
    up_az = np.deg2rad((land["aspect_deg"] + 180) % 360)  # aspect faces downslope; upslope is opposite
    flat = land["aspect_deg"] < 0
    ch_ft = land["ch_m"] / FT
    # active crown spread (Rothermel 1991): 3.34 x FM10 spread with 40 % of the 20-ft wind, no slope
    c10 = fuels.constants(10, m)
    phi10 = c10["C"] * (0.4 * u20_ftmin) ** c10["B"] * c10["rel"] ** -c10["E"]
    r_active = 3.34 * c10["r0"] * (1 + phi10) * FT  # m/min
    for code in np.unique(fuel):
        c = fuels.constants(code, m)
        idx = fuel == code
        if c is None:
            continue
        waf = fuels.waf(c["depth"], land["cc"][idx], ch_ft[idx])
        u_mid = waf * u20_ftmin
        phi_w = c["C"] * u_mid ** c["B"] * c["rel"] ** -c["E"]
        phi_s = c["slope_k"] * slope[idx] ** 2
        phi_s = np.where(flat[idx], 0, phi_s)
        # vector sum of wind and slope effects
        vx = phi_w * np.sin(to_az) + phi_s * np.sin(up_az[idx])
        vy = phi_w * np.cos(to_az) + phi_s * np.cos(up_az[idx])
        phi_e = np.hypot(vx, vy)
        # effective wind speed and its upper limit (Andrews et al. 2013)
        u_e = (phi_e * c["rel"] ** c["E"] / c["C"]) ** (1 / c["B"])
        u_lim = 96.8 * c["i_r"] ** (1 / 3)
        capped = u_e > u_lim
        phi_e = np.where(capped, c["C"] * u_lim ** c["B"] * c["rel"] ** -c["E"], phi_e)
        u_e = np.minimum(u_e, u_lim)
        r = c["r0"] * (1 + phi_e) * FT  # m/min
        i_b = c["i_r"] * c["t_r"] * (r / FT) / 60 * fuels.BTU_FT_S_TO_KW_M  # kW/m
        lb = fuels.length_to_breadth(u_e * FT / 60)
        e = np.sqrt(lb ** 2 - 1) / lb
        d = np.arctan2(vx, vy)
        # crown fire (Van Wagner 1977; Scott & Reinhardt 2001)
        cbh, cbd = land["cbh_m"][idx], land["cbd"][idx]
        forest = (land["cc"][idx] > 0) & (cbh > 0) & (cbd > 0)
        i0 = (0.010 * np.maximum(cbh, 0.1) * (460 + 25.9 * wx.foliar * 100)) ** 1.5
        torch = forest & (i_b >= i0)
        active = torch & (r_active >= 3.0 / np.maximum(cbd, 1e-3))
        cr = np.where(active, 2, np.where(torch, 1, 0)).astype(np.int8)
        r_fin = np.where(active, np.maximum(r, r_active), r)
        canopy_load = cbd * np.maximum(land["ch_m"][idx] - cbh, 0)  # kg/m2
        i_fin = np.where(active, i_b * r_fin / np.maximum(r, 1e-6) + 18000 * canopy_load * r_fin / 60, i_b)
        i_fin = np.where(torch & ~active, i_b + 18000 * canopy_load * r / 60, i_fin)
        lb_c = np.clip(1 + 0.125 * wx.wind_ms / 1.15 / MPH, 1, 8)  # crown-fire ellipse (Rothermel 1991)
        e = np.where(active, np.sqrt(lb_c ** 2 - 1) / lb_c, e)
        ros[idx], direc[idx], ecc[idx], fl[idx], crown[idx] = r_fin, d, e, fuels.flame_length_m(i_fin), cr
    return dict(ros=ros, dir=direc, ecc=ecc, fl=fl, crown=crown)


class Grid:
    """Static neighbour structure of a raster (edges between burnable cells)."""

    def __init__(self, burnable, cell_m):
        h, w = burnable.shape
        self.shape, self.cell = (h, w), cell_m
        rows, cols = np.nonzero(burnable)
        self.n = h * w
        src, dst, az, dist = [], [], [], []
        for dr, dc in OFFSETS:
            r2, c2 = rows + dr, cols + dc
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            r1, c1, r2, c2 = rows[ok], cols[ok], r2[ok], c2[ok]
            ok = burnable[r2, c2]
            if abs(dr) + abs(dc) == 3:  # knight move: both cells it passes between must burn too
                ok &= burnable[r1 + np.sign(dr) * (abs(dr) == 2), c1 + np.sign(dc) * (abs(dc) == 2)]
                ok &= burnable[r1 + dr - np.sign(dr) * (abs(dr) == 2), c1 + dc - np.sign(dc) * (abs(dc) == 2)]
            r1, c1, r2, c2 = r1[ok], c1[ok], r2[ok], c2[ok]
            src.append(r1 * w + c1); dst.append(r2 * w + c2)
            # azimuth clockwise from north; row index grows southwards
            az.append(np.full(len(r1), np.arctan2(dc, -dr))); dist.append(np.full(len(r1), np.hypot(dr, dc) * cell_m))
        self.src, self.dst = np.concatenate(src), np.concatenate(dst)
        self.az, self.dist = np.concatenate(az), np.concatenate(dist)

    def weights(self, fb):
        """Edge travel times (minutes) for a fire-behaviour field."""
        ros, d, e = fb["ros"].ravel(), fb["dir"].ravel(), fb["ecc"].ravel()

        def rate(cell):
            return ros[cell] * (1 - e[cell]) / (1 - e[cell] * np.cos(self.az - d[cell]))

        r1, r2 = rate(self.src), rate(self.dst)
        t = 0.5 * self.dist * (1 / np.maximum(r1, 1e-4) + 1 / np.maximum(r2, 1e-4))
        return csr_matrix((t, (self.src, self.dst)), shape=(self.n, self.n))

    def arrival(self, graph, ignitions, minutes):
        """Arrival time (minutes, inf where not reached) and predecessor of each cell, one row per ignition."""
        t, pred = dijkstra(graph, directed=True, indices=ignitions, limit=minutes, return_predecessors=True)
        return t, pred
