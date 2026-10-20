"""
Event-level research splits + time-verified pilot test set (Review 1019, Weeks 1-2).

Splits (whole fire events, never individual buildings):
  dev   : every event that contains one of the 50 legacy benchmark cases.  These
          cases were inspected repeatedly, used to write rules and to build the
          SFT targets, so the whole event is quarantined as a development set.
  test  : forward-in-time hold-out, events starting on/after TEST_START.
  val   : seeded ~20% of the remaining events.
  train : the rest.
Only events since 2019 are used (earlier DINS rarely recorded survivors), and
spatial near-duplicates are controlled: a test record within NEAR_M metres of a
dev/train/val record (same parcel burned twice, adjacent lots across events)
is flagged and excluded from the pilot sample.

Pilot test set (sampled from test events only):
  diagnostic : up to K_BAL destroyed + K_BAL survived per event (balanced)
  natural    : up to K_NAT per event, sampled at the event's true class ratio
Each sampled building gets a pre-fire crop from the latest Wayback release
before the fire start, with per-location capture date; samples whose imagery is
missing, post-dates the fire, is too stale, or predates construction are
rejected and counted (that acceptance rate is itself a deliverable).

Model-facing vs. label-side fields are separated explicitly:
  inputs.prefire      : structure type, year built, assessed value, location,
                        event name/date -- plausibly knowable before the fire
  inputs.inspector    : DINS building attributes (roof, eaves, vents, ...).
                        Recorded *after* the fire; missingness is outcome-
                        correlated (see dins_inventory.md).  Use only for the
                        oracle / ceiling conditions, never as a default input.
  labels              : damage, binary outcome, ignition location (if specific)

Outputs (output/splits/):
  event_splits.csv, record_splits.parquet, split_summary.json,
  pilot_test.jsonl, pilot_test_summary.md
Images: /data/wildfire_benchmark_data/wayback_crops/<GLOBALID>.jpg

Usage:
  python scripts/data/build_research_splits.py [--k-bal 15 --k-nat 20 --workers 6]
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402

SRC = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
DEV_CATALOG = ROOT / "SBFire_SJSU-20260730T030001Z-1-001/benchmark_results/dins_vlm_benchmark_50.json"
IMG_DIR = Path("/data/wildfire_benchmark_data/wayback_crops")
OUT = ROOT / "output" / "splits"

RESIDENTIAL = {"Single Residence", "Multiple Residence", "Mixed Commercial/Residential"}
DESTROYED = {"Destroyed (>50%)", "Major (25-50%)"}
SURVIVED = {"No Damage", "Affected (>0-10%)", "Minor (10-25%)"}
INSPECTOR_ATTRS = ["ROOFCONSTRUCTION", "EAVES", "VENTSCREEN", "EXTERIORSIDING", "WINDOWPANE",
                   "DECKPORCHONGRADE", "DECKPORCHELEVATED", "PATIOCOVERCARPORT",
                   "FENCEATTACHEDTOSTRUCTURE"]
UNKNOWN = {"", "unknown", "nan", "none", "not applicable", "<null>"}
NONSPECIFIC_IGNITION = UNKNOWN | {"fire", "a/b side", "b side"}
MIN_YEAR = 2019
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
VAL_FRACTION = 0.2
MIN_PER_CLASS_TEST = 30
NEAR_M = 100.0
SEED = 1019


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return None if s.lower() in UNKNOWN else s


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) or f <= 0 else f


def _seeded_rank(key: str) -> float:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def load_residential() -> pd.DataFrame:
    df = pd.read_parquet(SRC)
    df = df[df["STRUCTURECATEGORY"].isin(RESIDENTIAL) & df["DAMAGE"].isin(DESTROYED | SURVIVED)].copy()
    df["lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df[df["lat"].between(32, 42.1) & df["lon"].between(-124.6, -114)]
    df = df[df["INCIDENTSTARTDATE_ISO"].notna()]
    df["y"] = df["DAMAGE"].isin(DESTROYED).astype(int)
    df["event"] = (df["INCIDENTNAME"].fillna("NA").str.strip().str.title() + " | "
                   + df["INCIDENTSTARTDATE_ISO"].dt.strftime("%Y-%m-%d"))
    return df.reset_index(drop=True)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    dev_ids = {i["global_id"] for i in json.loads(DEV_CATALOG.read_text())["items"]}
    dev_events = set(df.loc[df["GLOBALID"].isin(dev_ids), "event"])
    ev = df.groupby("event").agg(start=("INCIDENTSTARTDATE_ISO", "first"), county=("COUNTY", "first"),
                                 n=("y", "size"), destroyed=("y", "sum")).reset_index()
    ev["survived"] = ev["n"] - ev["destroyed"]

    def split_of(r):
        if r["event"] in dev_events:
            return "dev"
        if r["start"].year < MIN_YEAR:
            return "excluded_pre2019"
        if r["start"] >= TEST_START:
            return "test"
        return "val" if _seeded_rank(r["event"]) < VAL_FRACTION else "train"

    ev["split"] = ev.apply(split_of, axis=1)
    return ev


def flag_spatial_neighbours(df: pd.DataFrame) -> pd.Series:
    """Distance (m) from each test record to the nearest non-test record (dev/train/val)."""
    other = df[df["split"].isin(["dev", "train", "val"])]
    test = df[df["split"] == "test"]
    tree = BallTree(np.radians(other[["lat", "lon"]].values), metric="haversine")
    d, _ = tree.query(np.radians(test[["lat", "lon"]].values), k=1)
    out = pd.Series(np.nan, index=df.index)
    out.loc[test.index] = d[:, 0] * 6_371_000.0
    return out


def sample_pilot(df: pd.DataFrame, ev: pd.DataFrame, k_bal: int, k_nat: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    elig = ev[(ev["split"] == "test") & (ev["destroyed"] >= MIN_PER_CLASS_TEST)
              & (ev["survived"] >= MIN_PER_CLASS_TEST)]["event"]
    pool = df[df["event"].isin(elig) & ~df["near_other_split"]]
    rows = []
    for e, g in pool.groupby("event"):
        for y in (0, 1):
            gy = g[g["y"] == y]
            take = gy.sample(n=min(k_bal, len(gy)), random_state=int(rng.integers(1 << 31)))
            rows.append(take.assign(pilot_set="diagnostic"))
        rest = g.drop(index=pd.concat(rows)[lambda x: x["event"] == e].index, errors="ignore")
        nat = rest.sample(n=min(k_nat, len(rest)), random_state=int(rng.integers(1 << 31)))
        rows.append(nat.assign(pilot_set="natural"))
    return pd.concat(rows).reset_index(drop=True)


def to_record(r: pd.Series, prov: dict) -> dict:
    ign = _clean(r["WHEREFIRESTARTEDONSTRUCTURE"])
    ign_specific = ign if ign and ign.lower() not in NONSPECIFIC_IGNITION else None
    yb = _num(r["YEARBUILT"])
    return {
        "global_id": r["GLOBALID"], "objectid": int(r["OBJECTID"]),
        "event": r["event"], "split": "test", "pilot_set": r["pilot_set"],
        "inputs": {
            "prefire": {
                "structure_type": _clean(r["STRUCTURETYPE"]),
                "year_built": int(yb) if yb and 1800 <= yb <= 2026 else None,
                "assessed_improved_value": _num(r["ASSESSEDIMPROVEDVALUE"]),
                "latitude": round(float(r["lat"]), 6), "longitude": round(float(r["lon"]), 6),
                "county": _clean(r["COUNTY"]), "incident_name": _clean(r["INCIDENTNAME"]),
                "incident_start": r["INCIDENTSTARTDATE_ISO"].date().isoformat(),
            },
            "inspector_postfire": {a.lower(): _clean(r[a]) for a in INSPECTOR_ATTRS},
        },
        "labels": {
            "damage": r["DAMAGE"], "outcome": "Destroyed" if r["y"] == 1 else "Survived",
            "ignition_location_specific": ign_specific,
            "defensive_actions": _clean(r["DEFENSIVEACTIONS"]),
        },
        "imagery": prov,
        "near_other_split_m": None if np.isnan(r["dist_other_m"]) else round(float(r["dist_other_m"]), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-bal", type=int, default=15, help="per class per test event (diagnostic set)")
    ap.add_argument("--k-nat", type=int, default=20, help="per test event at natural ratio")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-staleness-days", type=int, default=5 * 365)
    ap.add_argument("--skip-imagery", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_residential()
    ev = assign_splits(df)
    df = df.merge(ev[["event", "split"]], on="event", how="left")
    df["dist_other_m"] = flag_spatial_neighbours(df)
    df["near_other_split"] = df["dist_other_m"] < NEAR_M

    ev.assign(start=ev["start"].dt.date).sort_values(["split", "start"]).to_csv(OUT / "event_splits.csv", index=False)
    df[["GLOBALID", "OBJECTID", "event", "split", "y", "dist_other_m"]].to_parquet(OUT / "record_splits.parquet")
    summ = {s: {"events": int((ev["split"] == s).sum()), "records": int((df["split"] == s).sum()),
                "destroyed": int(df.loc[df["split"] == s, "y"].sum())}
            for s in ["dev", "train", "val", "test", "excluded_pre2019"]}
    summ["test_records_within_%dm_of_other_split" % NEAR_M] = int(df["near_other_split"].sum())
    summ["config"] = {"min_year": MIN_YEAR, "test_start": str(TEST_START.date()), "val_fraction": VAL_FRACTION,
                      "seed": SEED, "near_m": NEAR_M}
    (OUT / "split_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    pilot = sample_pilot(df, ev, args.k_bal, args.k_nat)
    print(f"[*] pilot sample: {len(pilot)} buildings from {pilot['event'].nunique()} test events")
    if args.skip_imagery:
        return

    releases = load_releases()

    def work(i):
        r = pilot.iloc[i]
        yb = _num(r["YEARBUILT"])
        prov = fetch_prefire_crop(
            float(r["lat"]), float(r["lon"]), r["INCIDENTSTARTDATE_ISO"].date(),
            IMG_DIR / f"{r['GLOBALID']}.jpg", releases,
            max_staleness_days=args.max_staleness_days,
            year_built=int(yb) if yb and 1800 <= yb <= 2026 else None)
        return to_record(r, prov.to_dict())

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(work, range(len(pilot))))

    with open(OUT / "pilot_test.jsonl", "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

    acc = [r for r in recs if r["imagery"]["accepted"]]
    reasons = Counter(r["imagery"]["reject_reason"].split(":")[0] for r in recs if not r["imagery"]["accepted"])
    stale = [r["imagery"]["staleness_days"] for r in acc]
    L = ["# Pilot test set (held-out events, time-verified imagery)\n",
         f"Built {dt.date.today().isoformat()} by `scripts/data/build_research_splits.py` (seed {SEED}).\n",
         "## Splits\n", "| Split | Events | Records | Destroyed |", "|---|---:|---:|---:|"]
    for s in ["dev", "train", "val", "test", "excluded_pre2019"]:
        L.append(f"| {s} | {summ[s]['events']} | {summ[s]['records']:,} | {summ[s]['destroyed']:,} |")
    L += [f"\nTest records within {NEAR_M:.0f} m of a dev/train/val record (excluded from pilot): "
          f"{summ['test_records_within_%dm_of_other_split' % NEAR_M]:,}\n",
          "## Imagery acceptance\n",
          f"Sampled {len(recs)}; accepted {len(acc)} ({len(acc) / max(len(recs), 1):.0%}).\n",
          "| Reject reason | Count |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    if stale:
        L.append(f"\nCapture-to-fire staleness (accepted): median {int(np.median(stale))} d, "
                 f"p90 {int(np.percentile(stale, 90))} d, max {max(stale)} d.\n")
    L += ["## Accepted samples per event\n",
          "| Event | Diagnostic D/S | Natural D/S | Specific ignition label |", "|---|---|---|---:|"]
    by = {}
    for r in acc:
        b = by.setdefault(r["event"], {"diagnostic": [0, 0], "natural": [0, 0], "ign": 0})
        b[r["pilot_set"]][0 if r["labels"]["outcome"] == "Destroyed" else 1] += 1
        b["ign"] += r["labels"]["ignition_location_specific"] is not None
    for e, b in sorted(by.items()):
        L.append(f"| {e.replace(' | ', ' (') + ')'} | {b['diagnostic'][0]}/{b['diagnostic'][1]} | {b['natural'][0]}/{b['natural'][1]} | {b['ign']} |")
    (OUT / "pilot_test_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
