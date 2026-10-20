"""
Full paginated download of the CAL FIRE DINS (POSTFIRE_MASTER_DATA_SHARE) layer.

Unlike the legacy downloader (single request, limit=200), this script pages
through every record ordered by OBJECTID, writes one raw JSON file per page
(so the run is resumable and auditable), and then merges everything into a
single Parquet table plus a provenance manifest (query URL, record count
reported by the server, download timestamp, SHA256 of each page).

Usage:
    python scripts/data/fetch_dins_full.py --out /data/wildfire_benchmark_data/calfire_dins/full
"""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query"
)
PAGE_SIZE = 2000
UA = {"User-Agent": "WildfireAI-DataAudit/1.0"}


def _get_json(params: dict, retries: int = 5) -> dict:
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            data = json.loads(raw)
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            return data, raw
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} retries: {last_err}")


def fetch_all(out_dir: Path) -> Path:
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    count_data, _ = _get_json({"where": "1=1", "returnCountOnly": "true", "f": "json"})
    total = int(count_data["count"])
    n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"[*] Server reports {total} records -> {n_pages} pages of {PAGE_SIZE}")

    manifest_pages = []
    for page in range(n_pages):
        page_file = pages_dir / f"page_{page:04d}.json"
        if page_file.exists() and page_file.stat().st_size > 1000:
            raw = page_file.read_bytes()
        else:
            params = {
                "where": "1=1",
                "outFields": "*",
                "orderByFields": "OBJECTID ASC",
                "resultOffset": page * PAGE_SIZE,
                "resultRecordCount": PAGE_SIZE,
                "returnGeometry": "false",
                "f": "json",
            }
            _, raw = _get_json(params)
            page_file.write_bytes(raw)
        n_feat = len(json.loads(raw).get("features", []))
        manifest_pages.append({
            "page": page,
            "file": page_file.name,
            "n_features": n_feat,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        print(f"    page {page + 1}/{n_pages}: {n_feat} features")

    rows = []
    for p in manifest_pages:
        data = json.loads((pages_dir / p["file"]).read_bytes())
        for feat in data.get("features", []):
            rows.append(feat.get("attributes", {}))

    import pandas as pd  # local import keeps the download path dependency-light

    df = pd.DataFrame(rows).drop_duplicates(subset=["OBJECTID"])
    for col in ("INCIDENTSTARTDATE",):
        if col in df.columns:
            df[col + "_ISO"] = pd.to_datetime(df[col], unit="ms", errors="coerce", utc=True)
    parquet_path = out_dir / "dins_full.parquet"
    df.to_parquet(parquet_path, index=False)

    manifest = {
        "source_layer": BASE_URL.rsplit("/query", 1)[0],
        "server_reported_count": total,
        "rows_after_dedup": int(len(df)),
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "page_size": PAGE_SIZE,
        "pages": manifest_pages,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] {len(df)} unique records -> {parquet_path}")
    return parquet_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/wildfire_benchmark_data/calfire_dins/full")
    args = ap.parse_args()
    fetch_all(Path(args.out))
