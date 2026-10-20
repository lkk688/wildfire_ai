"""
How much of the outcome is exposure vs. building-level vulnerability?

Tabular models (train/val events 2019-2023) scored on ALL test-event records
(2024+), reported three ways:
  overall AUROC          mixes between-event / between-neighbourhood exposure
  within-event AUROC     record-weighted mean of per-event AUROC
  near-pair accuracy     destroyed/survived pairs in the same event within R m;
                         P(score_destroyed > score_survived).  Neighbours share
                         roughly the same fire exposure, so this isolates the
                         building-level signal that imagery could add.
A spatial-only model (kNN outcome of *training* neighbours is impossible across
events), so we also report the pair accuracy of simple single features.

Output: output/experiments/exposure_control.{json,md}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.tabular_baselines_on_pilot import (  # noqa: E402
    GEOM, GEOM_COLS, INSPECTOR, SPLITS, SRC, model, prep)

OUT = ROOT / "output" / "experiments"
R_EARTH = 6_371_000.0
SEED = 1019


def near_pairs(test: pd.DataFrame, radius: float, max_pairs_per_destroyed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    pairs = []
    for _, g in test.groupby("event"):
        d_idx = np.where(g["y"].values == 1)[0]
        s_idx = np.where(g["y"].values == 0)[0]
        if not len(d_idx) or not len(s_idx):
            continue
        tree = BallTree(np.radians(g[["lat", "lon"]].values[s_idx]), metric="haversine")
        nb = tree.query_radius(np.radians(g[["lat", "lon"]].values[d_idx]), radius / R_EARTH)
        gi = g.index.values
        for di, ns in zip(d_idx, nb):
            if len(ns):
                for s in rng.choice(ns, size=min(max_pairs_per_destroyed, len(ns)), replace=False):
                    pairs.append((gi[di], gi[s_idx[s]]))
    return np.array(pairs)


def main() -> None:
    df = pd.read_parquet(SRC)
    df = prep(df.merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    df = df.merge(pd.read_parquet(GEOM), on="GLOBALID", how="left")
    df["log_nn_dist"] = np.log1p(df["nn_dist_m"])
    df["lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    train = df[df["split"].isin(["train", "val"])]
    test = df[df["split"] == "test"].reset_index(drop=True)
    pairs = {r: near_pairs(test, r) for r in (50, 100, 250)}
    num = ["YEARBUILT", "log_value"]
    rows = []
    for fs, cat, nm in [("prefire_meta", ["STRUCTURETYPE"], num),
                        ("prefire_meta+geometry", ["STRUCTURETYPE"], num + GEOM_COLS),
                        ("inspector (leaky oracle)", ["STRUCTURETYPE"] + INSPECTOR, num),
                        ("inspector+geometry (leaky oracle)", ["STRUCTURETYPE"] + INSPECTOR, num + GEOM_COLS)]:
        m = model("hgb", cat, nm).fit(train[cat + nm], train["y"])
        p = m.predict_proba(test[cat + nm])[:, 1]
        y = test["y"].values
        we = [(roc_auc_score(y[i], p[i]), len(i)) for _, i in test.groupby("event").indices.items()
              if len(set(y[i])) == 2 and min(y[i].sum(), len(i) - y[i].sum()) >= 10]
        r = {"features": fs, "n_test": int(len(test)), "overall_auroc": float(roc_auc_score(y, p)),
             "within_event_auroc": float(np.average([a for a, _ in we], weights=[n for _, n in we]))}
        for rad, pr in pairs.items():
            r[f"pair_acc_{rad}m"] = float(np.mean((p[pr[:, 0]] > p[pr[:, 1]]) + 0.5 * (p[pr[:, 0]] == p[pr[:, 1]])))
            r[f"n_pairs_{rad}m"] = int(len(pr))
        rows.append(r)
        print(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exposure_control.json").write_text(json.dumps(rows, indent=2))
    L = ["# Exposure control: overall vs within-event vs near-neighbour pairs\n",
         "HGB trained on 2019-2023 events, scored on all 2024+ test-event residential records. "
         "Pair accuracy = P(destroyed scored above a survived neighbour in the same event within R m); 0.5 = chance.\n",
         "| Features | Overall AUROC | Within-event AUROC | Pair acc ≤50 m | ≤100 m | ≤250 m |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['features']} | {r['overall_auroc']:.3f} | {r['within_event_auroc']:.3f} | "
                 f"{r['pair_acc_50m']:.3f} | {r['pair_acc_100m']:.3f} | {r['pair_acc_250m']:.3f} |")
    L.append(f"\nPairs: ≤50 m {rows[0]['n_pairs_50m']:,}; ≤100 m {rows[0]['n_pairs_100m']:,}; ≤250 m {rows[0]['n_pairs_250m']:,}.")
    (OUT / "exposure_control.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
