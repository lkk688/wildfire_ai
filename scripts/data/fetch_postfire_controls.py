"""
Positive-control imagery: earliest post-fire capture (<= 365 d) for the pilot
test buildings.  Answers "is building state perceivable in these crops at
all?" -- if a VLM cannot tell destroyed from survived on POST-fire imagery,
pre-fire failures say nothing about risk reasoning (resolution / geolocation
limits instead).  Never used as a pre-fire model input.

Output: output/splits/pilot_postfire.jsonl (pilot records + "imagery_postfire")
"""

import datetime as dt
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.imagery_provenance import fetch_postfire_crop, load_releases  # noqa: E402

SPL = ROOT / "output" / "splits"
IMG = Path("/data/wildfire_benchmark_data/wayback_crops_postfire")


def main() -> None:
    recs = [json.loads(line) for line in (SPL / "pilot_test.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r["imagery"]["accepted"] and r["pilot_set"] == "diagnostic"]
    rel = load_releases()

    def work(r):
        pf = r["inputs"]["prefire"]
        p = fetch_postfire_crop(pf["latitude"], pf["longitude"], dt.date.fromisoformat(pf["incident_start"]),
                                IMG / f"{r['global_id']}.jpg", rel)
        r["imagery_postfire"] = p.to_dict()
        return r

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(work, recs))
    with open(SPL / "pilot_postfire.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(Counter(r["imagery_postfire"]["reject_reason"] or "accepted" for r in out))


if __name__ == "__main__":
    main()
