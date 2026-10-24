"""
External (out-of-California) wildfire test set for post-fire damage recognition: the 2021 Marshall Fire (Colorado),
labels from BRIGHT (building_damage rasters: 0 background, 1 intact, 2 damaged, 3 destroyed; Chen et al. 2025),
imagery from Esri Wayback through the same capture-date-verified pipeline as the DINS sets
(BRIGHT's own post-event imagery is SAR, so it is not used).

Buildings = connected components of label > 0 in each BRIGHT tile (min area MIN_PX pixels). Label = majority class;
destroyed (3) -> "Destroyed", intact (1) -> "Survived"; components whose majority is "damaged" (2) are excluded
because that class does not map onto DINS levels. Components across tile borders are de-duplicated by centroid
(< DEDUP_M). Sanity check against the MTBS Marshall perimeter: 266/272 sampled destroyed pixels fall inside it.

Output: output/splits/postfire_ext_marshall.jsonl (same schema as postfire_test.jsonl), images under
/data/wildfire_benchmark_data/wayback_crops_postfire/ext_marshall/ and wayback_crops/ext_marshall/.
"""
import argparse
import datetime as dt
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.imagery_provenance import fetch_postfire_crop_fast, fetch_prefire_crop, load_releases  # noqa: E402

BRIGHT = Path("/data/wildfire_benchmark_data/bright/target")
POST_DIR = Path("/data/wildfire_benchmark_data/wayback_crops_postfire/ext_marshall")
PRE_DIR = Path("/data/wildfire_benchmark_data/wayback_crops/ext_marshall")
OUT = ROOT / "output" / "splits" / "postfire_ext_marshall.jsonl"
FIRE_START = dt.date(2021, 12, 30)
EVENT = "Marshall (CO, BRIGHT) | 2021-12-30"
MIN_PX, DEDUP_M = 60, 5.0


def buildings():
    out = []
    for f in sorted(glob.glob(str(BRIGHT / "marshall-wildfire_*_building_damage.tif"))):
        with rasterio.open(f) as s:
            a = s.read(1)
            T = Transformer.from_crs(s.crs, 4326, always_xy=True)
            lab, n = ndimage.label(a > 0)
            for i in range(1, n + 1):
                m = lab == i
                if m.sum() < MIN_PX:
                    continue
                vals, cnt = np.unique(a[m], return_counts=True)
                maj = int(vals[np.argmax(cnt)])
                rr, cc = np.nonzero(m)
                x, y = s.transform * (cc.mean() + 0.5, rr.mean() + 0.5)
                lon, lat = T.transform(x, y)
                out.append({"tile": Path(f).name, "lat": lat, "lon": lon, "label_bright": maj,
                            "area_m2": float(m.sum() * abs(s.res[0] * s.res[1]))})
    # de-duplicate across tiles
    keep = []
    for b in sorted(out, key=lambda r: -r["area_m2"]):
        if all(np.hypot((b["lon"] - k["lon"]) * 85000, (b["lat"] - k["lat"]) * 111000) > DEDUP_M for k in keep[-400:]):
            keep.append(b)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    bs = [b for b in buildings() if b["label_bright"] in (1, 3)]
    rng = np.random.default_rng(1019)
    sel = []
    for c in (1, 3):
        cls = [b for b in bs if b["label_bright"] == c]
        idx = rng.permutation(len(cls))[: a.max_per_class]
        sel += [cls[i] for i in idx]
    print(f"{len(bs)} buildings (intact/destroyed); sampled {len(sel)}", flush=True)
    rel = load_releases()
    POST_DIR.mkdir(parents=True, exist_ok=True)
    PRE_DIR.mkdir(parents=True, exist_ok=True)

    def work(ib):
        i, b = ib
        gid = f"marshall_{i:05d}"
        post = fetch_postfire_crop_fast(b["lat"], b["lon"], FIRE_START, POST_DIR / f"{gid}.jpg", rel)
        if not post.accepted:
            return None
        pre = fetch_prefire_crop(b["lat"], b["lon"], FIRE_START, PRE_DIR / f"{gid}.jpg", rel)
        outcome = "Destroyed" if b["label_bright"] == 3 else "Survived"
        return {"global_id": gid, "objectid": None, "event": EVENT, "split": "external", "pilot_set": "postfire_ext",
                "inputs": ["prefire"], "labels": {"damage": "Destroyed (>50%)" if outcome == "Destroyed" else "No Damage",
                                                  "outcome": outcome, "label_source": "BRIGHT building_damage",
                                                  "bright_class": b["label_bright"]},
                "lat": b["lat"], "lon": b["lon"], "bright_tile": b["tile"], "footprint_m2": b["area_m2"],
                "imagery": pre.to_dict(), "imagery_postfire": post.to_dict()}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = [r for r in ex.map(work, enumerate(sel)) if r]
    OUT.write_text("".join(json.dumps(r) + "\n" for r in recs))
    from collections import Counter
    print(f"[OK] {len(recs)}/{len(sel)} with post-fire capture; outcomes {Counter(r['labels']['outcome'] for r in recs)}; "
          f"pre accepted {sum(r['imagery']['accepted'] for r in recs)}; post capture dates "
          f"{Counter(r['imagery_postfire']['capture_date'] for r in recs).most_common(3)}")


if __name__ == "__main__":
    main()
