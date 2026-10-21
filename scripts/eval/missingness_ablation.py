"""
Exp 1 remainder (exploratory; test events): separate "what the attribute values say"
from "whether the attribute was recorded", on the SAME population (all 2024+ test
residential records), instead of comparing against a selected clean subset.

  raw            values with Unknown kept as its own category (as in earlier runs)
  values_imputed Unknown -> most frequent value in train (NOT leak-free: an all-mode
                 record is itself a missingness signature)
  values_random  Unknown -> random draw from the train distribution of observed values
                 (breaks the signature; lower bound on value signal on this population)
  missing_only   one indicator per attribute (Unknown or not) + count, no values
HGB fitted on train/val events.  Metrics: overall AUROC, within-event AUROC,
near-pair (<=100 m) accuracy.

Output: output/experiments/missingness_ablation.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.exposure_control_analysis import near_pairs  # noqa: E402
from scripts.eval.tabular_baselines_on_pilot import INSPECTOR, SPLITS, SRC, model, prep  # noqa: E402


def main() -> None:
    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    df["lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    tr_mask = df["split"].isin(["train", "val"])
    rng = np.random.default_rng(1019)
    for a in INSPECTOR:
        obs = df.loc[tr_mask & (df[a] != "UNKNOWN"), a]
        df[a + "__imp"] = df[a].where(df[a] != "UNKNOWN", obs.mode().iloc[0])
        miss = df[a] == "UNKNOWN"
        df[a + "__rnd"] = df[a]
        df.loc[miss, a + "__rnd"] = rng.choice(obs.values, size=int(miss.sum()))
        df[a + "__miss"] = (df[a] == "UNKNOWN").astype(int)
    df["n_missing"] = df[[a + "__miss" for a in INSPECTOR]].sum(axis=1)
    tr = df[tr_mask]
    te = df[df["split"] == "test"].reset_index(drop=True)
    pr = near_pairs(te, 100)
    y = te["y"].values
    sets = {
        "raw (Unknown as category)": (["STRUCTURETYPE"] + INSPECTOR, ["YEARBUILT", "log_value"]),
        "values_imputed (no missingness)": (["STRUCTURETYPE"] + [a + "__imp" for a in INSPECTOR], ["YEARBUILT", "log_value"]),
        "values_random_imputed (Unknown -> random draw from train values)":
            (["STRUCTURETYPE"] + [a + "__rnd" for a in INSPECTOR], ["YEARBUILT", "log_value"]),
        "missing_only (indicators, no values)": (["STRUCTURETYPE"], ["YEARBUILT", "log_value", "n_missing"]
                                                 + [a + "__miss" for a in INSPECTOR]),
        "prefire_meta only": (["STRUCTURETYPE"], ["YEARBUILT", "log_value"]),
    }
    L = ["# Missingness ablation (exploratory, same test population)\n",
         f"All {len(te):,} test-event residential records; {len(pr):,} near pairs (<=100 m). HGB fitted on train/val events.\n",
         "| Features | Overall AUROC | Within-event AUROC | Pair acc <=100 m |", "|---|---:|---:|---:|"]
    for name, (cat, num) in sets.items():
        m = model("hgb", cat, num).fit(tr[cat + num], tr["y"])
        p = m.predict_proba(te[cat + num])[:, 1]
        we = [(roc_auc_score(y[i], p[i]), len(i)) for _, i in te.groupby("event").indices.items()
              if min(y[i].sum(), len(i) - y[i].sum()) >= 10]
        wa = np.average([a for a, _ in we], weights=[n for _, n in we])
        pa = np.mean((p[pr[:, 0]] > p[pr[:, 1]]) + 0.5 * (p[pr[:, 0]] == p[pr[:, 1]]))
        L.append(f"| {name} | {roc_auc_score(y, p):.3f} | {wa:.3f} | {pa:.3f} |")
        print(L[-1])
    (ROOT / "output" / "experiments" / "missingness_ablation.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
