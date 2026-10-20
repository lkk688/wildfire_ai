"""
Objective, pre-fire-knowable layout features from the DINS structure points.

For every structure: distance to the nearest other structure in the same fire
event (any category: homes, outbuildings, commercial) and counts within 20 /
50 / 100 m.  Building locations are knowable before a fire, so these are fair
inputs; they are also an independent *verifier* for VLM spacing claims.

Caveat: DINS lists only structures inspected inside the damage-inspection area,
so counts near the inspection boundary are underestimates.  A building-
footprint layer (Overture / Microsoft) would be the cleaner source.

Output: output/features/geometry.parquet  (GLOBALID, nn_dist_m, n_20m, n_50m, n_100m)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
OUT = ROOT / "output" / "features"
R_EARTH = 6_371_000.0


def main() -> None:
    df = pd.read_parquet(SRC, columns=["GLOBALID", "INCIDENTNAME", "INCIDENTSTARTDATE_ISO", "LATITUDE", "LONGITUDE"])
    df["lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df[df["lat"].between(32, 42.1) & df["lon"].between(-124.6, -114)].copy()
    df["event"] = df["INCIDENTNAME"].fillna("NA").str.strip().str.title() + "|" + \
        df["INCIDENTSTARTDATE_ISO"].dt.strftime("%Y-%m-%d").fillna("NA")
    parts = []
    for _, g in df.groupby("event"):
        X = np.radians(g[["lat", "lon"]].values)
        tree = BallTree(X, metric="haversine")
        k = min(2, len(g))
        d, _ = tree.query(X, k=k)
        nn = d[:, 1] * R_EARTH if k == 2 else np.full(len(g), np.nan)
        out = {"GLOBALID": g["GLOBALID"].values, "nn_dist_m": nn}
        for r in (20, 50, 100):
            out[f"n_{r}m"] = tree.query_radius(X, r / R_EARTH, count_only=True) - 1
        parts.append(pd.DataFrame(out))
    res = pd.concat(parts)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "geometry.parquet")
    print(res.describe().round(1))


if __name__ == "__main__":
    main()
