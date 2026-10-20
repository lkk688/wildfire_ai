"""
Simple strong baselines on the held-out pilot test set (Review 1019, section 7).

Trained on the train+val events (2019-2023, dev events excluded), evaluated on
exactly the buildings in output/splits/pilot_test.jsonl, with the same metric
code as the VLM condition runs, so rows are directly comparable.

Feature sets
  prior           train-set class prior (constant)
  prefire_meta    structure type, year built, log assessed value  (what the VLM
                  "meta" condition sees, minus event/county names)
  inspector       + DINS post-fire inspector attributes, 'Unknown' kept as a
                  category (leaky oracle; compare with the VLM "inspector" run)

Output: output/experiments/tabular_baselines_pilot.{json,md}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot, _fmt, binary_metrics  # noqa: E402

SRC = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
SPLITS = ROOT / "output" / "splits" / "record_splits.parquet"
PILOT = ROOT / "output" / "splits" / "pilot_test.jsonl"
OUT = ROOT / "output" / "experiments"
INSPECTOR = ["ROOFCONSTRUCTION", "EAVES", "VENTSCREEN", "EXTERIORSIDING", "WINDOWPANE",
             "DECKPORCHONGRADE", "DECKPORCHELEVATED", "PATIOCOVERCARPORT", "FENCEATTACHEDTOSTRUCTURE"]
UNKNOWN = {"", "unknown", "nan", "none", "not applicable", "<null>"}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    for c in INSPECTOR + ["STRUCTURETYPE"]:
        s = df[c].astype("string").str.strip()
        df[c] = s.where(s.notna() & ~s.str.lower().isin(UNKNOWN), "UNKNOWN")
    yb = pd.to_numeric(df["YEARBUILT"], errors="coerce")
    df["YEARBUILT"] = yb.where(yb.between(1800, 2026))
    v = pd.to_numeric(df["ASSESSEDIMPROVEDVALUE"], errors="coerce")
    df["log_value"] = np.log1p(v.where(v > 0))
    return df


def model(kind: str, cat: list, num: list):
    if kind == "logreg":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), num)])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ("num", "passthrough", num)])
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, categorical_features=list(range(len(cat))), random_state=0))])


def main() -> None:
    df = pd.read_parquet(SRC)
    sp = pd.read_parquet(SPLITS)
    df = prep(df.merge(sp[["GLOBALID", "split", "y", "event"]], on="GLOBALID", how="inner"))
    train = df[df["split"].isin(["train", "val"])]
    pilot = [json.loads(line) for line in PILOT.read_text().splitlines()]
    pilot = [r for r in pilot if r["imagery"]["accepted"]]
    test = df.set_index("GLOBALID").loc[[r["global_id"] for r in pilot]].reset_index()
    test["pilot_set"] = [r["pilot_set"] for r in pilot]

    num = ["YEARBUILT", "log_value"]
    runs = []
    for fs, cat in [("prior", None), ("prefire_meta", ["STRUCTURETYPE"]),
                    ("inspector (leaky oracle)", ["STRUCTURETYPE"] + INSPECTOR)]:
        for kind in (["prior"] if cat is None else ["logreg", "hgb"]):
            if kind == "prior":
                p_all = np.full(len(test), train["y"].mean())
            else:
                m = model(kind, cat, num).fit(train[cat + num], train["y"])
                p_all = m.predict_proba(test[cat + num])[:, 1]
            for ps in ("diagnostic", "natural"):
                sel = (test["pilot_set"] == ps).values
                y, p, ev = test.loc[sel, "y"].values, p_all[sel], test.loc[sel, "event"].values
                r = binary_metrics(y, p)
                if kind != "prior":
                    r["auroc_ci95"] = _cluster_boot(ev, lambda i: binary_metrics(y[i], p[i])["auroc"])
                    r["FPR_ci95"] = _cluster_boot(ev, lambda i: binary_metrics(y[i], p[i])["FPR"])
                r.update({"feature_set": fs, "model": kind, "pilot_set": ps})
                runs.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tabular_baselines_pilot.json").write_text(json.dumps(
        {"train_records": int(len(train)), "train_events": int(train["event"].nunique()),
         "train_pos_rate": float(train["y"].mean()), "runs": runs}, indent=2))
    L = ["# Tabular baselines on the pilot test set\n",
         f"Trained on {len(train):,} residential records from {train['event'].nunique()} train/val events "
         f"(2019-2023, dev events excluded; positive rate {train['y'].mean():.1%}). Evaluated on the same "
         "time-verified pilot buildings as the VLM runs. Threshold 0.5 (note: train prior differs from the "
         "balanced diagnostic set, so threshold metrics are less informative than AUROC). "
         "95% CI = fire-event cluster bootstrap.\n",
         "| Pilot set | Features | Model | N | AUROC [95% CI] | Bal. acc | FPR [95% CI] | Brier |",
         "|---|---|---|---:|---|---:|---|---:|"]
    for r in runs:
        ci = r.get("auroc_ci95", (np.nan, np.nan))
        fci = r.get("FPR_ci95", (np.nan, np.nan))
        L.append(f"| {r['pilot_set']} | {r['feature_set']} | {r['model']} | {r['n']} | {_fmt(r['auroc'])} "
                 f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(r['balanced_accuracy'])} | {_fmt(r['FPR'], True)} "
                 f"[{_fmt(fci[0], True)}, {_fmt(fci[1], True)}] | {_fmt(r['brier'])} |")
    (OUT / "tabular_baselines_pilot.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
