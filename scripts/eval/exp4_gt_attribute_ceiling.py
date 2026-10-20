"""
Decisive experiment #4 (Review 1019): ground-truth attribute ceiling.

Question: if a model were given the inspector-recorded DINS building attributes
(i.e. a perfect perception module), how well could a simple model predict
Destroyed vs Survived on *unseen fire events*?  If this ceiling is low, training
a VLM to recover these attributes from imagery cannot help much.

Design
  * Residential DINS records, binary outcome, events since 2019 (survivors were
    rarely recorded before 2019 -> strong selection bias).
  * Grouped cross-validation by fire event (no event appears in train and test).
  * Feature sets
      prior          : class prior only
      attrs_raw      : categorical attributes with 'Unknown' kept as a category
                       (contains missingness leakage; reported for reference)
      attrs_observed : same model, restricted to records where every attribute
                       in CORE is actually observed (no missingness signal)
      attrs_obs_nomiss: attrs_observed model trained only on the observed
                       subset as well (clean)
      year_value     : YEARBUILT + ASSESSEDIMPROVEDVALUE only
  * Models: logistic regression (one-hot) and histogram gradient boosting.
  * Metrics: AUROC, balanced accuracy at 0.5, Brier; also mean per-event AUROC
    (ranking within an event, which removes between-event base-rate effects).

Outputs: output/experiments/gt_attribute_ceiling.{json,md}
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
OUT = ROOT / "output" / "experiments"

RESIDENTIAL = {"Single Residence", "Multiple Residence", "Mixed Commercial/Residential"}
DESTROYED = {"Destroyed (>50%)", "Major (25-50%)"}
SURVIVED = {"No Damage", "Affected (>0-10%)", "Minor (10-25%)"}
CORE = ["ROOFCONSTRUCTION", "EAVES", "VENTSCREEN", "EXTERIORSIDING", "WINDOWPANE",
        "DECKPORCHONGRADE", "DECKPORCHELEVATED", "FENCEATTACHEDTOSTRUCTURE"]
EXTRA_CAT = ["PATIOCOVERCARPORT", "STRUCTURETYPE"]
NUM = ["YEARBUILT", "ASSESSEDIMPROVEDVALUE"]
UNKNOWN = {"", "unknown", "nan", "none", "<null>"}
N_FOLDS = 5
MIN_EVENT_PER_CLASS = 10


def load() -> pd.DataFrame:
    df = pd.read_parquet(SRC)
    df = df[df["STRUCTURECATEGORY"].isin(RESIDENTIAL)]
    df = df[df["DAMAGE"].isin(DESTROYED | SURVIVED)].copy()
    df["y"] = df["DAMAGE"].isin(DESTROYED).astype(int)
    df["event"] = df["INCIDENTNAME"].fillna("NA").str.title() + "|" + \
        df["INCIDENTSTARTDATE_ISO"].dt.strftime("%Y-%m-%d").fillna("NA")
    df = df[df["INCIDENTSTARTDATE_ISO"].dt.year >= 2019]
    for c in CORE + EXTRA_CAT:
        s = df[c].astype("string").str.strip()
        df[c] = s.where(~s.str.lower().isin(UNKNOWN) & s.notna(), "UNKNOWN")
    for c in NUM:
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = v.where(v > 0)
    df["YEARBUILT"] = df["YEARBUILT"].where(df["YEARBUILT"].between(1800, 2026))
    df["log_value"] = np.log1p(df["ASSESSEDIMPROVEDVALUE"])
    # keep events with enough of both classes so per-event AUROC is defined
    cnt = df.groupby("event")["y"].agg(["sum", "count"])
    ok = cnt[(cnt["sum"] >= MIN_EVENT_PER_CLASS) & (cnt["count"] - cnt["sum"] >= MIN_EVENT_PER_CLASS)].index
    return df[df["event"].isin(ok)].reset_index(drop=True)


def make_model(kind: str, cat: list, num: list):
    if kind == "logreg":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
            ("num", Pipeline([("imp", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")),
                              ("sc", __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler())]), num),
        ], remainder="drop")
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ("num", "passthrough", num),
    ], remainder="drop")
    n_cat = len(cat)
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, categorical_features=list(range(n_cat)) if n_cat else None,
        random_state=0))])


def evaluate(df: pd.DataFrame, cat: list, num: list, kind: str) -> dict:
    gkf = GroupKFold(n_splits=N_FOLDS)
    p = np.zeros(len(df))
    for tr, te in gkf.split(df, df["y"], df["event"]):
        if kind == "prior":
            p[te] = df["y"].iloc[tr].mean()
            continue
        m = make_model(kind, cat, num)
        m.fit(df.iloc[tr][cat + num], df["y"].iloc[tr])
        p[te] = m.predict_proba(df.iloc[te][cat + num])[:, 1]
    y = df["y"].values
    per_event = []
    for ev, idx in df.groupby("event").indices.items():
        if len(set(y[idx])) == 2 and kind != "prior":
            per_event.append((ev, roc_auc_score(y[idx], p[idx]), len(idx)))
    w_auc = (np.average([a for _, a, _ in per_event], weights=[n for _, _, n in per_event])
             if per_event else float("nan"))
    return {
        "n": int(len(df)), "n_events": int(df["event"].nunique()),
        "pos_rate": round(float(y.mean()), 3),
        "auroc": round(float(roc_auc_score(y, p)), 4) if kind != "prior" else 0.5,
        "balanced_acc": round(float(balanced_accuracy_score(y, p >= 0.5)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "within_event_auroc_weighted": round(float(w_auc), 4) if per_event else None,
        "per_event": sorted([{"event": e, "auroc": round(float(a), 3), "n": int(n)} for e, a, n in per_event],
                            key=lambda r: -r["n"])[:15],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    observed_mask = (df[CORE] != "UNKNOWN").all(axis=1)
    df_obs = df[observed_mask].reset_index(drop=True)
    cnt = df_obs.groupby("event")["y"].agg(["sum", "count"])
    keep = cnt[(cnt["sum"] >= MIN_EVENT_PER_CLASS) & (cnt["count"] - cnt["sum"] >= MIN_EVENT_PER_CLASS)].index
    df_obs = df_obs[df_obs["event"].isin(keep)].reset_index(drop=True)
    num = ["YEARBUILT", "log_value"]

    runs = []
    for name, data, cat, nm, kinds in [
        ("prior", df, [], [], ["prior"]),
        ("attrs_raw (Unknown as category; leaky)", df, CORE + EXTRA_CAT, num, ["logreg", "hgb"]),
        ("year_value only", df, [], num, ["logreg", "hgb"]),
        ("prior (observed subset)", df_obs, [], [], ["prior"]),
        ("attrs_observed (all CORE observed; clean)", df_obs, CORE + EXTRA_CAT, num, ["logreg", "hgb"]),
        ("year_value only (observed subset)", df_obs, [], num, ["logreg", "hgb"]),
    ]:
        for k in kinds:
            r = evaluate(data, cat, nm, k)
            r.update({"feature_set": name, "model": k})
            runs.append(r)
            print(f"{name:<45}{k:<8} n={r['n']:<6} ev={r['n_events']:<3} AUROC={r['auroc']:<7} "
                  f"bAcc={r['balanced_acc']:<7} Brier={r['brier']:<7} withinEvAUROC={r['within_event_auroc_weighted']}")

    (OUT / "gt_attribute_ceiling.json").write_text(json.dumps({
        "design": "GroupKFold by fire event, residential DINS, events >= 2019, "
                  f">= {MIN_EVENT_PER_CLASS} per class per event",
        "observed_subset_fraction": round(float(observed_mask.mean()), 3),
        "runs": runs}, indent=2))

    L = ["# Experiment #4: Ground-Truth Attribute Ceiling (leave-event-out)\n",
         "Inspector-recorded DINS attributes -> Destroyed vs Survived. Grouped 5-fold CV by fire event; "
         "events since 2019 only (earlier DINS rarely recorded survivors). "
         f"Observed-subset = records where all {len(CORE)} core attributes are non-Unknown "
         f"({observed_mask.mean():.1%} of eligible records).\n",
         "Caveat: DINS attributes are recorded *after* the fire; for destroyed homes some are "
         "inferred from remains. This is an optimistic upper bound for any pre-fire perception module.\n",
         "| Feature set | Model | N | Events | Pos. rate | AUROC | Bal. acc | Brier | Within-event AUROC |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        L.append(f"| {r['feature_set']} | {r['model']} | {r['n']:,} | {r['n_events']} | {r['pos_rate']} | "
                 f"{r['auroc']} | {r['balanced_acc']} | {r['brier']} | {r['within_event_auroc_weighted']} |")
    (OUT / "gt_attribute_ceiling.md").write_text("\n".join(L) + "\n")
    print(f"[OK] {OUT / 'gt_attribute_ceiling.md'}")


if __name__ == "__main__":
    main()
