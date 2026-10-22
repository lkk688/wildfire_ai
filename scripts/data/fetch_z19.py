"""
(A2) Native-resolution crops: same Wayback release logic at zoom 19 (~0.25 m/px; 512 px ~ 75 m),
for the pre-fire train sample, T1 diagnostic buildings and T2 pair buildings.
Output: /data/wildfire_benchmark_data/wayback_crops_z19/<GLOBALID>.jpg + output/splits/z19_manifest.json
"""
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402

SPL = ROOT / "output" / "splits"
IMG = Path("/data/wildfire_benchmark_data/wayback_crops_z19")


def main():
    recs = {}
    for f in ("train_sample.jsonl", "pilot_test.jsonl"):
        for l in (SPL / f).read_text().splitlines():
            r = json.loads(l)
            if r["imagery"]["accepted"] and (f != "pilot_test.jsonl" or r["pilot_set"] == "diagnostic"):
                recs[r["global_id"]] = r
    for f in ("pairs_test.jsonl", "pairs_test_large.jsonl"):
        for l in (SPL / f).read_text().splitlines():
            q = json.loads(l)
            for s in ("destroyed", "survived"):
                recs[q[s]["global_id"]] = q[s]
    rel = load_releases()

    def work(r):
        pf = r["inputs"]["prefire"]
        p = fetch_prefire_crop(pf["latitude"], pf["longitude"], dt.date.fromisoformat(pf["incident_start"]),
                               IMG / f"{r['global_id']}.jpg", rel, zoom=19, year_built=pf.get("year_built"))
        return r["global_id"], p.to_dict()

    with ThreadPoolExecutor(max_workers=10) as ex:
        out = dict(ex.map(work, recs.values()))
    (SPL / "z19_manifest.json").write_text(json.dumps(out))
    print(len(out), sum(v["accepted"] for v in out.values()))


if __name__ == "__main__":
    main()
