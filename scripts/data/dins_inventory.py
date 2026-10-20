"""
DINS data-quality inventory (Review 1019, Week 1-2 deliverable B).

Reads the full DINS parquet produced by scripts/data/fetch_dins_full.py and
writes:
  output/data_inventory/dins_inventory.md        human-readable report
  output/data_inventory/dins_incidents.csv       per-incident table
  output/data_inventory/dins_inventory.json      machine-readable summary

Key questions answered:
  * How many residential records with a usable binary outcome exist?
  * How are they distributed across fire events and years?
  * Which attributes are missing, and does missingness correlate with the
    outcome (a leakage risk when attributes are fed to a model)?
  * How many records have a specific (non-Unknown) ignition-location label?
  * Which events are large enough, in both classes, to serve as held-out
    test events, and which fall inside the Esri Wayback era (>= 2014-02)?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
OUT = ROOT / "output" / "data_inventory"

RESIDENTIAL_CATEGORIES = {"Single Residence", "Multiple Residence", "Mixed Commercial/Residential"}
DESTROYED = {"Destroyed (>50%)", "Major (25-50%)"}
SURVIVED = {"No Damage", "Affected (>0-10%)", "Minor (10-25%)"}
ATTRS = [
    "ROOFCONSTRUCTION", "EAVES", "VENTSCREEN", "EXTERIORSIDING", "WINDOWPANE",
    "DECKPORCHONGRADE", "DECKPORCHELEVATED", "PATIOCOVERCARPORT",
    "FENCEATTACHEDTOSTRUCTURE", "PROPANETANKDISTANCE", "UTILITYMISCSTRUCTUREDISTANCE",
    "YEARBUILT", "ASSESSEDIMPROVEDVALUE", "DEFENSIVEACTIONS",
]
UNKNOWN_TOKENS = {"", "unknown", "nan", "none", "not applicable", "<null>"}
NONSPECIFIC_IGNITION = UNKNOWN_TOKENS | {"fire", "a/b side"}
WAYBACK_START = pd.Timestamp("2014-02-20", tz="UTC")
MIN_PER_CLASS_FOR_TEST_EVENT = 30


def is_missing(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = s.isna() | s.isin(UNKNOWN_TOKENS)
    if series.name in ("YEARBUILT", "ASSESSEDIMPROVEDVALUE"):
        num = pd.to_numeric(series, errors="coerce")
        out = out | num.isna() | (num <= 0)
    return out.fillna(True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)
    n_total = len(df)

    df["incident_key"] = (
        df["INCIDENTNAME"].fillna("UNKNOWN").str.strip().str.title()
        + " | " + df["INCIDENTSTARTDATE_ISO"].dt.strftime("%Y-%m-%d").fillna("no-date")
    )
    df["year"] = df["INCIDENTSTARTDATE_ISO"].dt.year

    res = df[df["STRUCTURECATEGORY"].isin(RESIDENTIAL_CATEGORIES)].copy()
    res["label"] = np.where(res["DAMAGE"].isin(DESTROYED), "Destroyed",
                            np.where(res["DAMAGE"].isin(SURVIVED), "Survived", "Excluded"))
    excluded = int((res["label"] == "Excluded").sum())
    res = res[res["label"] != "Excluded"]
    lat = pd.to_numeric(res["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(res["LONGITUDE"], errors="coerce")
    res["coord_ok"] = lat.between(32, 42.1) & lon.between(-124.6, -114)
    n_bad_coords = int((~res["coord_ok"]).sum())
    res = res[res["coord_ok"]]

    # ---- missingness vs outcome (leakage risk) ----
    miss_rows = []
    for a in ATTRS:
        m = is_missing(res[a])
        d = m[res["label"] == "Destroyed"].mean()
        s = m[res["label"] == "Survived"].mean()
        miss_rows.append({"attribute": a, "missing_destroyed": round(float(d), 3),
                          "missing_survived": round(float(s), 3),
                          "gap": round(float(d - s), 3)})
    miss_df = pd.DataFrame(miss_rows).sort_values("gap", key=lambda x: -x.abs())

    # ---- ignition-location label coverage ----
    ign = res["WHEREFIRESTARTEDONSTRUCTURE"].astype("string").str.strip().str.lower()
    res["ignition_specific"] = ~(ign.isna() | ign.isin(NONSPECIFIC_IGNITION))
    ign_cov = res.groupby("label")["ignition_specific"].agg(["sum", "mean"]).reset_index()
    ign_vocab = (res.loc[res["ignition_specific"], "WHEREFIRESTARTEDONSTRUCTURE"]
                 .value_counts().to_dict())

    # ---- per-incident table ----
    g = res.groupby("incident_key")
    inc = pd.DataFrame({
        "incident": g["INCIDENTNAME"].first(),
        "start_date": g["INCIDENTSTARTDATE_ISO"].min().dt.strftime("%Y-%m-%d"),
        "county": g["COUNTY"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        "n_residential": g.size(),
        "n_destroyed": g["label"].apply(lambda x: int((x == "Destroyed").sum())),
        "n_survived": g["label"].apply(lambda x: int((x == "Survived").sum())),
        "n_ignition_specific": g["ignition_specific"].sum().astype(int),
        "wayback_era": g["INCIDENTSTARTDATE_ISO"].min() >= WAYBACK_START,
    }).reset_index()
    inc["destroyed_frac"] = (inc["n_destroyed"] / inc["n_residential"]).round(3)
    inc["test_event_candidate"] = (
        (inc["n_destroyed"] >= MIN_PER_CLASS_FOR_TEST_EVENT)
        & (inc["n_survived"] >= MIN_PER_CLASS_FOR_TEST_EVENT)
        & inc["wayback_era"]
    )
    inc = inc.sort_values("n_residential", ascending=False)
    inc.to_csv(OUT / "dins_incidents.csv", index=False)

    by_year = res.groupby(["year", "label"]).size().unstack(fill_value=0)
    cand = inc[inc["test_event_candidate"]]
    top_share = float(inc["n_residential"].head(5).sum() / inc["n_residential"].sum())

    summary = {
        "source": str(SRC),
        "n_total_records": n_total,
        "n_residential_binary_labelled": int(len(res)),
        "n_excluded_inaccessible": excluded,
        "n_bad_coordinates_dropped": n_bad_coords,
        "label_counts": res["label"].value_counts().to_dict(),
        "raw_damage_counts_residential": res["DAMAGE"].value_counts().to_dict(),
        "n_incidents": int(inc.shape[0]),
        "n_incidents_wayback_era": int(inc["wayback_era"].sum()),
        "n_test_event_candidates": int(cand.shape[0]),
        "records_in_test_event_candidates": int(cand["n_residential"].sum()),
        "top5_incident_share": round(top_share, 3),
        "ignition_specific_by_label": {r["label"]: {"n": int(r["sum"]), "frac": round(float(r["mean"]), 4)}
                                       for _, r in ign_cov.iterrows()},
        "ignition_specific_vocab": {str(k): int(v) for k, v in ign_vocab.items()},
        "missingness_vs_label": miss_rows,
    }
    (OUT / "dins_inventory.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- markdown report ----
    L = []
    L.append("# DINS Data-Quality Inventory\n")
    L.append(f"Source: `{SRC}` (full layer, {n_total:,} records; see `manifest.json` for page hashes).\n")
    L.append("## 1. Usable residential records\n")
    L.append("| Item | Count |\n|---|---:|")
    L.append(f"| All records | {n_total:,} |")
    L.append(f"| Residential (Single/Multiple/Mixed) with binary label | {len(res):,} |")
    for k, v in res["label"].value_counts().items():
        L.append(f"| - {k} | {v:,} |")
    L.append(f"| Excluded (Inaccessible) | {excluded:,} |")
    L.append(f"| Dropped (coordinates outside CA bbox / missing) | {n_bad_coords:,} |")
    L.append(f"| Distinct incidents (name + start date) | {inc.shape[0]:,} |")
    L.append(f"| Incidents in Wayback era (start >= 2014-02-20) | {int(inc['wayback_era'].sum()):,} |")
    L.append(f"| Share of residential records in top-5 incidents | {top_share:.1%} |\n")
    L.append("Binary mapping: Destroyed = Destroyed(>50%) + Major(25-50%); "
             "Survived = No Damage + Affected(>0-10%) + Minor(10-25%).\n")

    L.append("## 2. Outcome by year\n")
    L.append("| Year | Destroyed | Survived |\n|---|---:|---:|")
    for y, row in by_year.iterrows():
        L.append(f"| {int(y) if pd.notna(y) else 'n/a'} | {row.get('Destroyed', 0):,} | {row.get('Survived', 0):,} |")
    L.append("")

    L.append("## 3. Ignition-location label coverage\n")
    L.append("`WHEREFIRESTARTEDONSTRUCTURE` excluding blank / Unknown / Not Applicable / 'Fire'.\n")
    L.append("| Label | Records with specific location | Fraction |\n|---|---:|---:|")
    for _, r in ign_cov.iterrows():
        L.append(f"| {r['label']} | {int(r['sum']):,} | {r['mean']:.2%} |")
    L.append("\nVocabulary: " + ", ".join(f"{k} ({v})" for k, v in list(ign_vocab.items())[:15]) + "\n")

    L.append("## 4. Attribute missingness vs outcome (leakage risk)\n")
    L.append("If an attribute is far more often 'Unknown' for destroyed homes, then feeding the raw "
             "field (including the literal 'Unknown') to a model leaks the label: inspectors cannot "
             "observe the vents of a house that burned down.\n")
    L.append("| Attribute | Missing (Destroyed) | Missing (Survived) | Gap |\n|---|---:|---:|---:|")
    for _, r in miss_df.iterrows():
        L.append(f"| {r['attribute']} | {r['missing_destroyed']:.1%} | {r['missing_survived']:.1%} | {r['gap']:+.1%} |")
    L.append("")

    L.append(f"## 5. Held-out test-event candidates (>= {MIN_PER_CLASS_FOR_TEST_EVENT} per class, Wayback era)\n")
    L.append(f"{cand.shape[0]} incidents, {int(cand['n_residential'].sum()):,} residential records.\n")
    L.append("| Incident | Start | County | N | Destroyed | Survived | Ignition-specific |\n|---|---|---|---:|---:|---:|---:|")
    for _, r in cand.head(40).iterrows():
        L.append(f"| {r['incident']} | {r['start_date']} | {r['county']} | {r['n_residential']:,} | "
                 f"{r['n_destroyed']:,} | {r['n_survived']:,} | {r['n_ignition_specific']:,} |")
    L.append("\nFull per-incident table: `dins_incidents.csv`.\n")
    (OUT / "dins_inventory.md").write_text("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\n[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
