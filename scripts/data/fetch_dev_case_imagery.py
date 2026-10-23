"""
Verified imagery for the 50-case VLM dev set shown in the web UI (VLM benchmark tab).

For every case we fetch, from Esri World Imagery Wayback, with per-location
capture dates read from each release's metadata layer (never a fallback to the
current basemap):
  * prefire  - latest release before the incident start whose capture date is
               before the start (src.vlm_benchmark.imagery_provenance.fetch_prefire_crop,
               default staleness limit, rejects captures older than YEARBUILT)
  * postfire - earliest capture after the start, within 365 days
               (fetch_postfire_crop_fast)
  * current  - the newest release in the catalogue, with its capture date at
               this location (fetch_prefire_crop with "today" as the cut-off and
               no staleness limit)

Incident start dates come from the CAL FIRE DINS records (matched by OBJECTID).

Outputs (in the benchmark images dir / benchmark dir):
  images/case_<id>_{prefire,postfire,current}_verified.jpg
  dev_case_imagery_provenance.json   {case_id: {epoch: ImageryProvenance.to_dict()}}

Usage: python scripts/data/fetch_dev_case_imagery.py [--workers 3] [--force]
"""

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.imagery_provenance import (  # noqa: E402
    fetch_postfire_crop_fast,
    fetch_prefire_crop,
    load_releases,
)
from src.vlm_benchmark.service import BENCHMARK_DIR, BENCHMARK_JSON, IMAGES_DIR  # noqa: E402

DINS_PARQUET = Path("/data/wildfire_benchmark_data/calfire_dins/full/dins_full.parquet")
OUT_JSON = BENCHMARK_DIR / "dev_case_imagery_provenance.json"


def load_cases():
    items = json.loads(BENCHMARK_JSON.read_text())["items"]
    dins = pd.read_parquet(DINS_PARQUET, columns=["OBJECTID", "INCIDENTSTARTDATE_ISO", "YEARBUILT"])
    dins = dins.drop_duplicates("OBJECTID").set_index("OBJECTID")
    cases = []
    for it in items:
        oid = int(it["id"])
        if oid not in dins.index:
            print(f"case {oid}: not found in DINS parquet, skipped")
            continue
        start = pd.Timestamp(dins.loc[oid, "INCIDENTSTARTDATE_ISO"]).date()
        yb = it.get("year_built") or dins.loc[oid, "YEARBUILT"]
        try:
            yb = int(yb) if yb and int(yb) > 1800 else None
        except (TypeError, ValueError):
            yb = None
        cases.append({"id": str(oid), "lat": float(it["latitude"]), "lon": float(it["longitude"]),
                      "fire_start": start, "year_built": yb})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="refetch cases already in the provenance file")
    args = ap.parse_args()

    releases = load_releases()
    today = dt.date.today()
    prev = json.loads(OUT_JSON.read_text()).get("cases", {}) if OUT_JSON.exists() and not args.force else {}
    cases = load_cases()

    def work(c):
        cid = c["id"]
        if cid in prev and all(k in prev[cid] for k in ("prefire", "postfire", "current")):
            return cid, prev[cid]
        rec = {"fire_start": c["fire_start"].isoformat(), "fire_start_source": "CAL FIRE DINS INCIDENTSTARTDATE"}
        rec["prefire"] = fetch_prefire_crop(
            c["lat"], c["lon"], c["fire_start"], IMAGES_DIR / f"case_{cid}_prefire_verified.jpg", releases,
            year_built=c["year_built"]).to_dict()
        rec["postfire"] = fetch_postfire_crop_fast(
            c["lat"], c["lon"], c["fire_start"], IMAGES_DIR / f"case_{cid}_postfire_verified.jpg", releases,
            max_lag_days=365).to_dict()
        cur = fetch_prefire_crop(
            c["lat"], c["lon"], today, IMAGES_DIR / f"case_{cid}_current_verified.jpg", releases,
            max_staleness_days=None).to_dict()
        cur["fire_start"] = c["fire_start"].isoformat()  # fetch used today's date as the cut-off
        cur["cutoff_date"] = today.isoformat()
        rec["current"] = cur
        print(cid, {k: (rec[k]["capture_date"] if rec[k]["accepted"] else rec[k]["reject_reason"])
                    for k in ("prefire", "postfire", "current")}, flush=True)
        return cid, rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = dict(ex.map(work, cases))

    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "Esri World Imagery Wayback; capture dates from each release's metadata layer (SRC_DATE)",
        "cases": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    for ep in ("prefire", "postfire", "current"):
        print(ep, Counter(r[ep]["reject_reason"] or "accepted" for r in results.values()))
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
