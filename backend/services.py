"""
Backend data services for the wildfire map view.

Every value served here comes from a real source: the Bellwether GeoTIFFs shared with the pilot partners (local),
public live services (see backend/live_sources.py), or the Microsoft building footprints extracted locally.
Nothing is synthesised; when a source is unavailable or does not cover an area, the response says so.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend import live_sources as live
from src.data_fetchers import crop_bellwether_by_bbox

BASE_DIR = Path(__file__).resolve().parent.parent
REF_DATA_DIR = BASE_DIR / "SBFire_SJSU-20260730T030001Z-1-001" / "SBFire_SJSU" / "ref_data"
STATIC_DIR = BASE_DIR / "backend" / "static"
NUMBERS_TEX = BASE_DIR / "paper" / "numbers.tex"

# [min_lat, min_lng, max_lat, max_lng]
REGION_BOUNDS = {
    "san_bruno": {"name": "San Bruno WUI (Peninsula)", "center": [37.618, -122.425], "bounds": [37.58, -122.46, 37.65, -122.39]},
    "san_jose": {"name": "San Jose Foothills (Santa Clara)", "center": [37.338, -121.886], "bounds": [37.25, -121.98, 37.42, -121.78]},
    "santa_cruz": {"name": "Santa Cruz Mountains (CZU WUI)", "center": [37.125, -122.050], "bounds": [37.02, -122.18, 37.22, -121.92]},
}
BW_CATS = ["Very Low", "Low", "Moderate", "Significant", "High", "Very High", "Extreme"]
BW_CUTOFFS_1YR = [0.00004, 0.00020, 0.00100, 0.00400, 0.00667, 0.01333]
FHSZ_CODES = {0: "Not in FHSZ", 1: "Moderate", 2: "High", 3: "Very High"}


def region_bbox(region: str) -> Tuple[float, float, float, float]:
    b = REGION_BOUNDS.get(region, REGION_BOUNDS["san_bruno"])["bounds"]
    return (b[1], b[0], b[3], b[2])  # (min_lon, min_lat, max_lon, max_lat)


def parse_bbox(bbox: Optional[str], region: str) -> Tuple[float, float, float, float]:
    if bbox:
        x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
        return (x0, y0, x1, y1)
    return region_bbox(region)


def _bw_path(is_5_year: bool) -> Path:
    for f in REF_DATA_DIR.glob("prediction_for_2026*.tif"):
        if ("2031" in f.name) == is_5_year:
            return f
    raise FileNotFoundError("Bellwether prediction GeoTIFF not found in ref_data")


class GISDataService:
    def __init__(self):
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        self.ref_data_dir = REF_DATA_DIR

    # ------------------------------------------------------------------ Bellwether (real partner raster)
    def bellwether_coverage(self) -> Optional[Tuple[float, float, float, float]]:
        try:
            import rasterio
            with rasterio.open(_bw_path(False)) as src:
                b = src.bounds
            return (b.left, b.bottom, b.right, b.top)
        except Exception:  # noqa: BLE001
            return None

    def get_bellwether_overlay(self, is_5_year: bool = False, region: str = "san_bruno", min_lat=None, min_lng=None,
                               max_lat=None, max_lng=None) -> Dict:
        if None not in (min_lat, min_lng, max_lat, max_lng):
            bb = (min_lng, min_lat, max_lng, max_lat)
        else:
            bb = region_bbox(region)
        cov = self.bellwether_coverage()
        if cov is None:
            return {"covered": False, "reason": "Bellwether GeoTIFFs are not available on this server."}
        if bb[0] >= cov[2] or bb[2] <= cov[0] or bb[1] >= cov[3] or bb[3] <= cov[1]:
            return {"covered": False, "coverage_bounds": [[cov[1], cov[0]], [cov[3], cov[2]]],
                    "reason": "The Bellwether rasters shared with this pilot cover the San Mateo / Santa Cruz area only."}
        layer = crop_bellwether_by_bbox(bb[1], bb[0], bb[3], bb[2], is_5_year=is_5_year)
        layer["covered"] = True
        try:
            import rasterio
            from rasterio.windows import from_bounds
            with rasterio.open(_bw_path(is_5_year)) as src:
                ib = (max(bb[0], cov[0]), max(bb[1], cov[1]), min(bb[2], cov[2]), min(bb[3], cov[3]))
                a = src.read(1, window=from_bounds(*ib, transform=src.transform).round_offsets().round_lengths())
            layer["valid_share_pct"] = round(float(np.mean(~np.isnan(a) & (a >= 0))) * 100, 1)
        except Exception:  # noqa: BLE001
            layer["valid_share_pct"] = None
        layer["window"] = "2026-04-01 to 2027-04-01" if not is_5_year else "2026-04-01 to 2031-04-01"
        layer["source"] = "Bellwether wildfire hazard model (X), partner GeoTIFF release"
        return layer

    def bellwether_at(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            import rasterio
            out = {}
            for five, key in ((False, "1yr"), (True, "5yr")):
                with rasterio.open(_bw_path(five)) as src:
                    b = src.bounds
                    if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                        return None
                    v = float(next(src.sample([(lon, lat)]))[0])
                out[key] = None if (np.isnan(v) or v < 0) else v
            if out["1yr"] is None:
                return None
            cat = BW_CATS[int(np.searchsorted(BW_CUTOFFS_1YR, out["1yr"], side="right"))]
            return {"prob_1yr": out["1yr"], "prob_5yr": out["5yr"], "category_1yr": cat,
                    "source": "Bellwether 100 m GeoTIFF (2026-04 release)"}
        except Exception:  # noqa: BLE001
            return None

    def get_risk_factors(self, region: Optional[str] = None, is_5_year: bool = False) -> Dict:
        """Bellwether risk-factor COG: per cell, the top-5 contributing (positive weight) and top-5 mitigating
        (negative weight) features. Aggregated over the region: each feature's share of the total contributing
        (or mitigating) weight, and the share of cells where it appears in the top 5."""
        tag = "5" if is_5_year else "1"
        mapping_path = self.ref_data_dir / f"feature_mapping_{tag}_year.json"
        cog_path = self.ref_data_dir / f"risk_factors_{tag}_year_cog.tif"
        if not mapping_path.exists() or not cog_path.exists():
            return {"available": False, "reason": "Risk-factor files not found"}
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(cog_path) as src:
            if region:
                bb = region_bbox(region)
                b = src.bounds
                ib = (max(bb[0], b.left), max(bb[1], b.bottom), min(bb[2], b.right), min(bb[3], b.top))
                if ib[0] >= ib[2] or ib[1] >= ib[3]:
                    return {"available": False, "reason": "No risk-factor coverage in this region"}
                bands = src.read(window=from_bounds(*ib, transform=src.transform).round_offsets().round_lengths())
            else:
                bands = src.read()
            desc = src.descriptions
        out = {}
        for kind in ("contributing", "mitigating"):
            name_idx = [i for i, d in enumerate(desc) if d and kind in d and d.endswith("feature_name")]
            weight_idx = [desc.index(desc[i].replace("_name", "_weight")) for i in name_idx]
            sums: Dict[str, float] = {}
            cells: Dict[str, int] = {}
            n_cells = 0
            for ni, wi in zip(name_idx, weight_idx):
                ids, w = bands[ni], bands[wi]
                ok = (ids != -9999) & (w != -9999) & ~np.isnan(w)
                n_cells = max(n_cells, int(ok.sum()))
                for fid, val in zip(ids[ok].astype(int), np.abs(w[ok])):
                    name = mapping.get(str(fid), f"Feature {fid}")
                    sums[name] = sums.get(name, 0.0) + float(val)
                    cells[name] = cells.get(name, 0) + 1
            total = sum(sums.values()) or 1.0
            out[kind] = [{"feature": k, "weight_share_pct": round(v / total * 100, 1),
                          "cells_in_top5_pct": round(cells[k] / max(n_cells, 1) * 100, 1)}
                         for k, v in sorted(sums.items(), key=lambda x: -x[1])[:6]]
        return {"available": True, "model": f"{tag}-year", **out, "region": region,
                "source": "Bellwether risk-factor GeoTIFF (top-5 contributing and mitigating features per 100 m cell)"}

    # ------------------------------------------------------------------ live layers
    def get_perimeters(self, bbox) -> Dict:
        out, errors = {}, {}
        # current perimeters and incidents statewide (so active fires anywhere in California are visible);
        # historical perimeters only for the requested area (the statewide archive is large)
        for key, fn in (("current", lambda: live.current_perimeters(live.CA_BBOX)),
                        ("historic", lambda: live.historic_perimeters(bbox)),
                        ("incidents", lambda: live.calfire_incidents(live.CA_BBOX))):
            try:
                out[key] = fn()
            except Exception as e:  # noqa: BLE001
                out[key] = {"type": "FeatureCollection", "features": []}
                errors[key] = str(e)[:200]
        out["errors"] = errors
        return out

    def get_weather(self, lat: float, lon: float) -> Dict:
        w = live.current_weather(lat, lon)
        try:
            w["alerts"] = live.nws_alerts(lat, lon)
        except Exception as e:  # noqa: BLE001
            w["alerts"] = []
            w["alerts_error"] = str(e)[:200]
        w["red_flag_warning"] = any("Red Flag" in (a.get("event") or "") for a in w["alerts"])
        return w

    # ------------------------------------------------------------------ point query (map click)
    def query_point(self, lat: float, lng: float, radius_feet: float = 130.0) -> Dict:
        """Everything the real sources say about a clicked location. Each part fails independently."""
        radius_m = radius_feet * 0.3048
        res: Dict = {"center": {"lat": lat, "lng": lng}, "radius_feet": radius_feet, "radius_meters": round(radius_m, 1),
                     "errors": {}}

        def part(name, fn):
            try:
                res[name] = fn()
            except Exception as e:  # noqa: BLE001
                res[name] = None
                res["errors"][name] = f"{type(e).__name__}: {str(e)[:160]}"

        part("live_weather", lambda: self.get_weather(lat, lng))
        part("bellwether", lambda: self.bellwether_at(lat, lng))
        part("fhsz", lambda: live.fhsz_at(lat, lng))
        part("land_cover", lambda: live.land_cover_around(lat, lng, max(radius_m, 50.0)))
        part("historic_fires", lambda: live.historic_fires_at(lat, lng))
        part("buildings_in_radius", lambda: live.buildings_within(lat, lng, radius_m))
        part("terrain_slope_share_pct", lambda: live.slope_at(lat, lng))

        def nearest_station():
            fc = live.fuel_moisture((lng - 0.6, lat - 0.6, lng + 0.6, lat + 0.6))
            best = None
            for f in fc["features"]:
                lo, la = f["geometry"]["coordinates"]
                d = math.hypot((lo - lng) * 111320 * math.cos(math.radians(lat)), (la - lat) * 110574) / 1000
                if f["properties"].get("fuel_moisture_10h_pct") is not None and (best is None or d < best["distance_km"]):
                    best = {**f["properties"], "distance_km": round(d, 1)}
            return best
        part("nearest_fuel_station", nearest_station)

        w = res.get("live_weather") or {}
        if w.get("wind_speed_mph") is not None:
            # A simple, stated geometric heuristic (not a fire-behaviour model): stretch the radius downwind by
            # (1 + wind_mph / 20). The corridor drawn on the map uses this major axis.
            res["wind_adjusted_major_radius_meters"] = round(radius_m * (1.0 + w["wind_speed_mph"] / 20.0), 1)
            res["corridor_note"] = ("Downwind stretch = radius x (1 + wind mph / 20): a display heuristic, "
                                    "not a fire-spread model.")
        return res

    # ------------------------------------------------------------------ catalog and status
    def get_data_catalog(self) -> List[Dict]:
        return live.SOURCES

    def check_sources(self, region: str) -> List[Dict]:
        c = REGION_BOUNDS.get(region, REGION_BOUNDS["san_bruno"])["center"]
        rows = live.check_all(region_bbox(region), c[0], c[1])
        cov = self.bellwether_coverage()
        for r in rows:
            if r["id"] == "bellwether":
                r["status"] = ({"ok": True, "sample": f"coverage lon {cov[0]:.2f}..{cov[2]:.2f}, lat {cov[1]:.2f}..{cov[3]:.2f}"}
                               if cov else {"ok": False, "error": "GeoTIFFs not found"})
        return rows

    # ------------------------------------------------------------------ model comparison (computed from real layers)
    def get_model_evaluation_data(self, region: str = "san_bruno") -> Dict:
        bb = region_bbox(region)
        out: Dict = {"region": region, "region_name": REGION_BOUNDS.get(region, {}).get("name", region),
                     "models": MODEL_CARDS, "comparison_matrix": COMPARISON_MATRIX, "guidance": GUIDANCE,
                     "research_evidence": self.research_evidence()}
        try:
            out["analysis"] = self._bellwether_vs_fhsz(bb)
        except Exception as e:  # noqa: BLE001
            out["analysis"] = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        return out

    def _bellwether_vs_fhsz(self, bb) -> Dict:
        import rasterio
        from rasterio.features import rasterize
        from rasterio.windows import from_bounds
        from shapely.geometry import shape
        cov = self.bellwether_coverage()
        if cov is None:
            return {"available": False, "reason": "Bellwether GeoTIFFs not available."}
        ib = (max(bb[0], cov[0]), max(bb[1], cov[1]), min(bb[2], cov[2]), min(bb[3], cov[3]))
        if ib[0] >= ib[2] or ib[1] >= ib[3]:
            return {"available": False, "reason": "No Bellwether coverage in this region (the shared rasters cover "
                                                  "the San Mateo / Santa Cruz pilot area only)."}
        with rasterio.open(_bw_path(False)) as src:
            win = from_bounds(*ib, transform=src.transform).round_offsets().round_lengths()
            p = src.read(1, window=win).astype(float)
            tr = src.window_transform(win)
        valid = ~np.isnan(p) & (p >= 0)
        cat = np.searchsorted(BW_CUTOFFS_1YR, np.nan_to_num(p, nan=0), side="right")
        zones = live.fhsz(ib)
        code_of = {"Moderate": 1, "High": 2, "Very High": 3}
        shapes = sorted(((shape(f["geometry"]), code_of.get(f["properties"]["zone"], 0)) for f in zones["features"]
                         if f.get("geometry")), key=lambda x: x[1])  # higher zones drawn last
        z = rasterize(shapes, out_shape=p.shape, transform=tr, fill=0, dtype="uint8") if shapes else np.zeros(p.shape, "uint8")
        hist = live.historic_perimeters(ib, since_year=2000)
        hshapes = [(shape(f["geometry"]), 1) for f in hist["features"] if f.get("geometry")]
        burned = (rasterize(hshapes, out_shape=p.shape, transform=tr, fill=0, dtype="uint8") if hshapes
                  else np.zeros(p.shape, "uint8")).astype(bool)
        n = int(valid.sum())
        if n == 0:
            return {"available": False, "reason": "The Bellwether raster has no values in this region."}
        crosstab = []
        for zc, zname in FHSZ_CODES.items():
            m = valid & (z == zc)
            if not m.any():
                continue
            crosstab.append({"fhsz": zname, "cells": int(m.sum()), "area_pct": round(m.sum() / n * 100, 1),
                             "bellwether_1yr_mean_pct": round(float(p[m].mean()) * 100, 3),
                             "bellwether_1yr_median_pct": round(float(np.median(p[m])) * 100, 3),
                             "share_high_or_above_pct": round(float((cat[m] >= 4).mean()) * 100, 1),
                             "burned_since_2000_pct": round(float(burned[m].mean()) * 100, 1),
                             "bellwether_category_pct": {BW_CATS[k]: round(float((cat[m] == k).mean()) * 100, 1)
                                                         for k in range(7)}})
        under = valid & (cat >= 4) & (z <= 1)   # Bellwether High+ where FHSZ is none / Moderate
        over = valid & (z == 3) & (cat <= 1)    # FHSZ Very High where Bellwether is Low or Very Low
        cases = self._largest_areas(under, p, z, tr, "Bellwether High or above, FHSZ none or Moderate") + \
            self._largest_areas(over, p, z, tr, "FHSZ Very High, Bellwether Low or Very Low")
        return {"available": True, "analysed_bbox": ib, "cells": n, "cell_size_m": 100,
                "bellwether_valid_share_pct": round(n / p.size * 100, 1),
                "bellwether_window": "2026-04-01 to 2027-04-01 (1-year model)",
                "fhsz_sources": list(live.FHSZ_LAYERS.keys()),
                "crosstab": crosstab,
                "rank_correlation_spearman": self._spearman(p[valid], z[valid]),
                "disagreement_area_pct": {"bellwether_high_fhsz_low": round(float(under.sum()) / n * 100, 1),
                                          "fhsz_very_high_bellwether_low": round(float(over.sum()) / n * 100, 1)},
                "disagreement_areas": cases,
                "note": ("FHSZ maps long-term hazard (expected fire behaviour); Bellwether gives the probability of a "
                         "cell burning in the next year. Disagreement shows where they differ, not which is right.")}

    @staticmethod
    def _spearman(a, b) -> Optional[float]:
        from scipy.stats import spearmanr
        if len(a) < 10 or len(set(b.tolist())) < 2:
            return None
        return round(float(spearmanr(a, b).correlation), 3)

    def _largest_areas(self, mask, p, z, tr, label, k: int = 3) -> List[Dict]:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n == 0:
            return []
        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        out = []
        for idx in np.argsort(-sizes)[:k]:
            m = lab == idx + 1
            rr, cc = np.nonzero(m)
            r0, c0 = rr.mean(), cc.mean()
            lon, lat = tr * (c0 + 0.5, r0 + 0.5)
            case = {"type": label, "lat": round(lat, 5), "lon": round(lon, 5), "area_ha": int(m.sum()),
                    "bellwether_1yr_mean_pct": round(float(p[m].mean()) * 100, 3),
                    "fhsz": FHSZ_CODES[int(np.bincount(z[m]).argmax())]}
            try:
                case["past_fires"] = live.historic_fires_at(lat, lon)[:3]
            except Exception:  # noqa: BLE001
                case["past_fires"] = None
            try:
                case["land_cover"] = live.land_cover_around(lat, lon, 150)["within_radius_pct"]
            except Exception:  # noqa: BLE001
                case["land_cover"] = None
            try:
                case["buildings_within_300m"] = live.buildings_within(lat, lon, 300)["count"]
            except Exception:  # noqa: BLE001
                case["buildings_within_300m"] = None
            out.append(case)
        return out

    def research_evidence(self) -> Dict:
        """Held-out results from this project's own evaluation (paper/numbers.tex, generated from output/)."""
        if not NUMBERS_TEX.exists():
            return {"available": False}
        m = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", NUMBERS_TEX.read_text()))
        g = lambda k: m.get(k)  # noqa: E731
        return {"available": True, "source": "paper/numbers.tex (generated from output/ by scripts/paper/make_tables.py)",
                "items": [
                    {"task": "Pre-fire: which single home will be destroyed (2024-25 test fires)",
                     "result": (f"Claude Sonnet AUROC {g('preClaudeSonnetAuc')}; frozen DINOv2 probe {g('refDinoAuc')}; "
                               f"building-metadata model {g('refMetaAuc')}"),
                     "meaning": "Weak: aerial images before a fire carry little building-level signal."},
                    {"task": "Pre-fire: same-fire neighbours 15-100 m apart (exposure controlled)",
                     "result": (f"Sonnet pair accuracy {g('pairClaudeSonnetImg')} on test fires, {g('pairValClaudeSonnetImg')} on the "
                               f"pre-registered replication; metadata-only baseline {g('refMetaPair')}"),
                     "meaning": "Near chance once fire exposure is controlled."},
                    {"task": "Post-fire: recognise destroyed homes",
                     "result": f"Sonnet {g('posctlSonnet')} (positive control); Qwen3.5-4B + LoRA {g('pfSftFour')}; frozen DINOv2 probe {g('pfLp')} AUROC",
                     "meaning": "Strong: damage is visible after the fire."},
                    {"task": "Effect of a random (not by-fire) split",
                     "result": f"pre-fire AUROC {g('splitMetaEvent')} by fire vs {g('splitMetaRandom')} random",
                     "meaning": "Random splits inflate apparent skill."},
                ]}


# ---------------------------------------------------------------------- descriptive content (factual, sourced)
MODEL_CARDS = [
    {"id": "bellwether", "name": "Bellwether wildfire hazard model", "developer": "Bellwether, X (Alphabet)",
     "type": "Machine-learning landscape hazard model", "spatial_resolution": "100 m",
     "update_cadence": "Quarterly", "output_metric": "Probability of a cell being inside a wildfire perimeter (1 or 5 years)",
     "core_methodology": "Trained on hundreds of regularly updated vegetation, weather, terrain and human-presence "
                         "variables; outputs are calibrated to observed fire frequencies (Bellwether release docs).",
     "strengths": ["Forward-looking probability at 100 m", "Updated quarterly as conditions change"],
     "limitations": ["Not a fire-spread or real-time model", "Does not model smoke damage or structure vulnerability",
                     "Scores change between versioned releases"],
     "color": "#F97316", "badge": "Machine learning", "in_this_app": "Map overlay and comparison (pilot area only)"},
    {"id": "calfire_fhsz", "name": "CAL FIRE Fire Hazard Severity Zones", "developer": "CAL FIRE",
     "type": "Regulatory hazard classification", "spatial_resolution": "Zone polygons",
     "update_cadence": "Periodic map revisions (SRA 2023, LRA 2025)", "output_metric": "Moderate / High / Very High",
     "core_methodology": "Hazard based on expected fire behaviour and burn probability over decades, from fuels, "
                         "terrain and weather.",
     "strengths": ["Legal basis for building standards and disclosure", "Statewide and consistent"],
     "limitations": ["Measures hazard, not the chance of burning next year", "Updated infrequently"],
     "color": "#EF4444", "badge": "Regulatory", "in_this_app": "Map overlay and comparison (official services)"},
    {"id": "insurance", "name": "Commercial insurance catastrophe models", "developer": "Private vendors",
     "type": "Actuarial risk scores", "spatial_resolution": "Parcel level", "update_cadence": "Varies by vendor",
     "output_metric": "Proprietary scores", "core_methodology": "Proprietary; not available to this app.",
     "strengths": ["Used directly in underwriting"], "limitations": ["Methods and scores are not public"],
     "color": "#8B5CF6", "badge": "Proprietary", "in_this_app": "Not shown: no real data available"},
]
COMPARISON_MATRIX = [
    {"dimension": "Question answered", "bellwether": "Will this 100 m cell burn in the next 1 or 5 years?",
     "calfire": "How severe is the long-term wildfire hazard here?", "insurance": "What is the expected loss for this policy?"},
    {"dimension": "Resolution", "bellwether": "100 m raster", "calfire": "Zone polygons", "insurance": "Parcel"},
    {"dimension": "Updates", "bellwether": "Quarterly", "calfire": "Map revisions every several years",
     "insurance": "Vendor-specific"},
    {"dimension": "Structure vulnerability", "bellwether": "Not modelled", "calfire": "Not modelled (drives building code)",
     "insurance": "Varies"},
    {"dimension": "Access", "bellwether": "Partner GeoTIFFs", "calfire": "Public", "insurance": "Proprietary"},
]
GUIDANCE = {
    "for_fire_marshals": [
        "Use FHSZ for code and disclosure obligations, and Bellwether's near-term probability to prioritise inspections "
        "and fuel work where the two disagree.",
        "Check current conditions on the map (HRRR wind, RAWS fuel moisture, NWS Red Flag Warnings) before field decisions.",
    ],
    "for_homeowners": [
        "California's 'Safer from Wildfires' regulation requires admitted insurers to recognise listed mitigation "
        "measures in pricing (CDI).",
        "Home hardening and defensible space (Zones 0-2) follow CAL FIRE and Board of Forestry guidance.",
    ],
}
