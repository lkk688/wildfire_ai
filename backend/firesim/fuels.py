"""
Surface and crown fire behaviour for the Scott & Burgan (2005) 40 fuel models.

Rothermel (1972) surface spread with Albini (1976) changes as described in Andrews (2018, RMRS-GTR-371), dynamic
herbaceous curing, the effective-wind limit of Andrews et al. (2013), Byram intensity and flame length, and Van Wagner
(1977) / Rothermel (1991) crown fire as combined by Scott & Reinhardt (2001). US customary units inside, SI outside.
"""
from functools import lru_cache

import numpy as np

TPA = 0.0459137  # tons/acre -> lb/ft2
FT_MIN_TO_M_MIN = 0.3048
BTU_FT_S_TO_KW_M = 3.46141

# code: (name, dynamic, [1h, 10h, 100h, herb, woody] loads tons/acre, [sav 1h, herb, woody] 1/ft, depth ft, Mx dead %)
FBFM40 = {
    101: ("GR1", 1, [0.10, 0.00, 0.00, 0.30, 0.00], [2200, 2000, 9999], 0.4, 15),
    102: ("GR2", 1, [0.10, 0.00, 0.00, 1.00, 0.00], [2000, 1800, 9999], 1.0, 15),
    103: ("GR3", 1, [0.10, 0.40, 0.00, 1.50, 0.00], [1500, 1300, 9999], 2.0, 30),
    104: ("GR4", 1, [0.25, 0.00, 0.00, 1.90, 0.00], [2000, 1800, 9999], 2.0, 15),
    105: ("GR5", 1, [0.40, 0.00, 0.00, 2.50, 0.00], [1800, 1600, 9999], 1.5, 40),
    106: ("GR6", 1, [0.10, 0.00, 0.00, 3.40, 0.00], [2200, 2000, 9999], 1.5, 40),
    107: ("GR7", 1, [1.00, 0.00, 0.00, 5.40, 0.00], [2000, 1800, 9999], 3.0, 15),
    108: ("GR8", 1, [0.50, 1.00, 0.00, 7.30, 0.00], [1500, 1300, 9999], 4.0, 30),
    109: ("GR9", 1, [1.00, 1.00, 0.00, 9.00, 0.00], [1800, 1600, 9999], 5.0, 40),
    121: ("GS1", 1, [0.20, 0.00, 0.00, 0.50, 0.65], [2000, 1800, 1800], 0.9, 15),
    122: ("GS2", 1, [0.50, 0.50, 0.00, 0.60, 1.00], [2000, 1800, 1800], 1.5, 15),
    123: ("GS3", 1, [0.30, 0.25, 0.00, 1.45, 1.25], [1800, 1600, 1600], 1.8, 40),
    124: ("GS4", 1, [1.90, 0.30, 0.10, 3.40, 7.10], [1800, 1600, 1600], 2.1, 40),
    141: ("SH1", 1, [0.25, 0.25, 0.00, 0.15, 1.30], [2000, 1800, 1600], 1.0, 15),
    142: ("SH2", 0, [1.35, 2.40, 0.75, 0.00, 3.85], [2000, 9999, 1600], 1.0, 15),
    143: ("SH3", 0, [0.45, 3.00, 0.00, 0.00, 6.20], [1600, 9999, 1400], 2.4, 40),
    144: ("SH4", 0, [0.85, 1.15, 0.20, 0.00, 2.55], [2000, 1800, 1600], 3.0, 30),
    145: ("SH5", 0, [3.60, 2.10, 0.00, 0.00, 2.90], [750, 9999, 1600], 6.0, 15),
    146: ("SH6", 0, [2.90, 1.45, 0.00, 0.00, 1.40], [750, 9999, 1600], 2.0, 30),
    147: ("SH7", 0, [3.50, 5.30, 2.20, 0.00, 3.40], [750, 9999, 1600], 6.0, 15),
    148: ("SH8", 0, [2.05, 3.40, 0.85, 0.00, 4.35], [750, 9999, 1600], 3.0, 40),
    149: ("SH9", 1, [4.50, 2.45, 0.00, 1.55, 7.00], [750, 1800, 1500], 4.4, 40),
    161: ("TU1", 1, [0.20, 0.90, 1.50, 0.20, 0.90], [2000, 1800, 1600], 0.6, 20),
    162: ("TU2", 0, [0.95, 1.80, 1.25, 0.00, 0.20], [2000, 9999, 1600], 1.0, 30),
    163: ("TU3", 1, [1.10, 0.15, 0.25, 0.65, 1.10], [1800, 1600, 1400], 1.3, 30),
    164: ("TU4", 0, [4.50, 0.00, 0.00, 0.00, 2.00], [2300, 9999, 2000], 0.5, 12),
    165: ("TU5", 0, [4.00, 4.00, 3.00, 0.00, 3.00], [1500, 9999, 750], 1.0, 25),
    181: ("TL1", 0, [1.00, 2.20, 3.60, 0.00, 0.00], [2000, 9999, 9999], 0.2, 30),
    182: ("TL2", 0, [1.40, 2.30, 2.20, 0.00, 0.00], [2000, 9999, 9999], 0.2, 25),
    183: ("TL3", 0, [0.50, 2.20, 2.80, 0.00, 0.00], [2000, 9999, 9999], 0.3, 20),
    184: ("TL4", 0, [0.50, 1.50, 4.20, 0.00, 0.00], [2000, 9999, 9999], 0.4, 25),
    185: ("TL5", 0, [1.15, 2.50, 4.40, 0.00, 0.00], [2000, 9999, 1600], 0.6, 25),
    186: ("TL6", 0, [2.40, 1.20, 1.20, 0.00, 0.00], [2000, 9999, 9999], 0.3, 25),
    187: ("TL7", 0, [0.30, 1.40, 8.10, 0.00, 0.00], [2000, 9999, 9999], 0.4, 25),
    188: ("TL8", 0, [5.80, 1.40, 1.10, 0.00, 0.00], [1800, 9999, 9999], 0.3, 35),
    189: ("TL9", 0, [6.65, 3.30, 4.15, 0.00, 0.00], [1800, 9999, 1600], 0.6, 35),
    201: ("SB1", 0, [1.50, 3.00, 11.00, 0.00, 0.00], [2000, 9999, 9999], 1.0, 25),
    202: ("SB2", 0, [4.50, 4.25, 4.00, 0.00, 0.00], [2000, 9999, 9999], 1.0, 25),
    203: ("SB3", 0, [5.50, 2.75, 3.00, 0.00, 0.00], [2000, 9999, 9999], 1.2, 25),
    204: ("SB4", 0, [5.25, 3.50, 5.25, 0.00, 0.00], [2000, 9999, 9999], 2.7, 25),
}
# Anderson (1982) model 10 (timber litter and understory): used for active crown-fire spread (Rothermel 1991)
FM10 = ("FM10", 0, [3.01, 2.00, 5.01, 0.00, 2.00], [2000, 9999, 1500], 1.0, 25)
NONBURNABLE = {91, 92, 93, 98, 99}


def name(code):
    return FBFM40[code][0] if code in FBFM40 else f"NB{code - 90}" if code in NONBURNABLE else str(code)


def _particles(model, m):
    """Particles (load lb/ft2, sav 1/ft, moisture fraction, is_live) after dynamic herb transfer.
    m: dict with d1, d10, d100, herb, woody moisture as fractions."""
    _, dyn, loads, sav, _, _ = model
    w = [x * TPA for x in loads]
    herb = w[3]
    dead_herb = 0.0
    if dyn and herb > 0:
        cured = float(np.clip(-1.11 * m["herb"] + 1.33, 0, 1))
        dead_herb, herb = herb * cured, herb * (1 - cured)
    dead = [(w[0], sav[0], m["d1"]), (w[1], 109.0, m["d10"]), (w[2], 30.0, m["d100"]), (dead_herb, sav[1], m["d1"])]
    live = [(herb, sav[1], m["herb"]), (w[4], sav[2], m["woody"])]
    return [p for p in dead if p[0] > 0], [p for p in live if p[0] > 0]


def _eta_m(mf, mx):
    r = min(1.0, mf / mx) if mx > 0 else 1.0
    return max(0.0, 1 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3)


def surface_constants(model, m):
    """Moisture-dependent, wind-free part of Rothermel for one fuel model. Returns dict or None if nothing burns."""
    rho_p, s_t, eta_s, h = 32.0, 0.0555, 0.174 * 0.01 ** -0.19, 8000.0
    depth, mx_dead = model[4], model[5] / 100
    dead, live = _particles(model, m)
    if not dead and not live:
        return None
    a = {k: np.array([p[1] * p[0] / rho_p for p in parts]) for k, parts in (("d", dead), ("l", live))}
    a_d, a_l = a["d"].sum(), a["l"].sum()
    a_t = a_d + a_l
    out = {}
    cat = {}
    for k, parts, ak in (("d", dead, a_d), ("l", live, a_l)):
        if not parts:
            continue
        f = a[k] / ak
        w = np.array([p[0] for p in parts]); s = np.array([p[1] for p in parts]); mo = np.array([p[2] for p in parts])
        cat[k] = dict(f=f, w=w, s=s, m=mo, wn=(f * w * (1 - s_t)).sum(), sigma=(f * s).sum(), mf=(f * mo).sum(),
                      F=ak / a_t)
    sigma = sum(c["F"] * c["sigma"] for c in cat.values())
    rho_b = sum(c["w"].sum() for c in cat.values()) / depth
    beta = rho_b / rho_p
    beta_op = 3.348 * sigma ** -0.8189
    rel = beta / beta_op
    g_max = sigma ** 1.5 / (495 + 0.0594 * sigma ** 1.5)
    A = 133 * sigma ** -0.7913
    gamma = g_max * rel ** A * np.exp(A * (1 - rel))
    # moisture of extinction for live fuel (Rothermel 1972, Albini 1976)
    mx_live = mx_dead
    if "l" in cat and "d" in cat:
        d, l = cat["d"], cat["l"]
        wd = (d["w"] * np.exp(-138 / d["s"])).sum()
        wl = (l["w"] * np.exp(-500 / l["s"])).sum()
        mfd = (d["w"] * np.exp(-138 / d["s"]) * d["m"]).sum() / wd if wd > 0 else 0
        mx_live = max(mx_dead, 2.9 * (wd / wl) * (1 - mfd / mx_dead) - 0.226) if wl > 0 else mx_dead
    i_r = 0.0
    for k, c in cat.items():
        i_r += c["wn"] * h * _eta_m(c["mf"], mx_dead if k == "d" else mx_live) * eta_s
    i_r *= gamma
    xi = np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192 + 0.2595 * sigma)
    heat_sink = rho_b * sum(c["F"] * (c["f"] * np.exp(-138 / c["s"]) * (250 + 1116 * c["m"])).sum() for c in cat.values())
    if i_r <= 0 or heat_sink <= 0:
        return None
    out.update(r0=i_r * xi / heat_sink, i_r=i_r, sigma=sigma, beta=beta, rel=rel, depth=depth,
               C=7.47 * np.exp(-0.133 * sigma ** 0.55), B=0.02526 * sigma ** 0.54, E=0.715 * np.exp(-3.59e-4 * sigma),
               slope_k=5.275 * beta ** -0.3, t_r=384 / sigma)
    return out


@lru_cache(maxsize=4096)
def _cached(code, d1, d10, d100, herb, woody):
    model = FM10 if code == 10 else FBFM40.get(code)
    if model is None:
        return None
    return surface_constants(model, dict(d1=d1, d10=d10, d100=d100, herb=herb, woody=woody))


def constants(code, m):
    r = lambda x: round(float(x), 3)  # noqa: E731
    return _cached(int(code), r(m["d1"]), r(m["d10"]), r(m["d100"]), r(m["herb"]), r(m["woody"]))


def waf(depth_ft, cc_pct, ch_ft):
    """Wind adjustment factor, 20-ft wind to midflame (Albini & Baughman 1979; Andrews 2012)."""
    depth_ft = np.maximum(depth_ft, 0.1)
    unsh = 1.83 / np.log((20 + 0.36 * depth_ft) / (0.13 * depth_ft))
    ch = np.maximum(ch_ft, 1.0)
    f = np.clip(cc_pct / 300.0, 1e-3, 1)
    sh = 0.555 / (np.sqrt(f * ch) * np.log((20 + 0.36 * ch) / (0.13 * ch)))
    return np.where((cc_pct > 5) & (ch_ft > 0), np.minimum(sh, unsh), unsh)


def length_to_breadth(u_mid_ms):
    """Anderson (1983) fire ellipse length-to-breadth ratio from (effective) midflame wind in m/s, capped at 8."""
    lb = 0.936 * np.exp(0.2566 * u_mid_ms) + 0.461 * np.exp(-0.1548 * u_mid_ms) - 0.397
    return np.clip(lb, 1.0, 8.0)


def flame_length_m(i_kw_m):
    """Byram (1959) flame length from fireline intensity."""
    return 0.0775 * np.maximum(i_kw_m, 0) ** 0.46
