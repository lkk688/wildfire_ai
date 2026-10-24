"""
(V2) Defensible-space features from pre-fire imagery, using the OpenEarthMap segmenter
(scripts/train/train_oem_segmenter.py, output/models/oem_segmenter/best.pt).

For every building in the pre-fire sets (T1 single buildings, T2 neighbour pairs, train and replication sets) the
capture-date-verified Wayback crop (z18, 512 px, ~0.47-0.50 m/px, centred on the DINS point) is segmented into
8 classes. The home = the building-class component closest to the crop centre (within 15 m); rings are measured
outward from its edge:  0-2 m | 2-10 m (~Zone 1) | 10-30 m (~Zone 2) | 30-75 m.
Per ring: share of each class; plus distance to the nearest other building, number of other building components
within 30 m, home footprint area, and whether a home component was found.

Only pixels inside the crop are counted; ring shares are normalised by the ring pixels inside the crop.
Output: output/experiments/defensible_space/features.parquet (+ seg masks in seg/<global_id>.png).
Run in the GPU env.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.train_oem_segmenter import CLASSES, Segmenter, norm  # noqa: E402

SPL = ROOT / "output" / "splits"
OUT = ROOT / "output" / "experiments" / "defensible_space"
SINGLE = ["train_sample", "pilot_test", "test_natural_xl", "confirm_direct"]
PAIRS = ["train_pairs", "pairs_test", "pairs_test_large", "pairs_test_xl", "confirm_pairs"]
RINGS = [(0, 2), (2, 10), (10, 30), (30, 75)]
BUILDING = CLASSES.index("building")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def collect():
    recs = {}
    for f in SINGLE:
        for r in jl(SPL / f"{f}.jsonl"):
            if r["imagery"].get("accepted"):
                recs[r["global_id"]] = r
    for f in PAIRS:
        for q in jl(SPL / f"{f}.jsonl"):
            for s in ("destroyed", "survived"):
                r = q[s]
                if r["imagery"].get("accepted"):
                    recs[r["global_id"]] = r
    return recs


def lat_of(r):
    if r.get("lat") is not None:
        return float(r["lat"])
    try:
        return float(r["inputs"]["prefire"]["latitude"])
    except Exception:  # noqa: BLE001
        return 37.0


def features(seg, mpp):
    h, w = seg.shape
    cy, cx = h / 2, w / 2
    bl, n = ndimage.label(seg == BUILDING)
    out = {"home_found": 0}
    home = None
    if n:
        yy, xx = np.mgrid[0:h, 0:w]
        dist_c = np.hypot(yy + 0.5 - cy, xx + 0.5 - cx)  # pixels from the DINS point (crop centre)
        best = None
        for i in range(1, n + 1):
            d = dist_c[bl == i].min() * mpp
            if d <= 15 and (best is None or d < best[0]):
                best = (d, i)
        if best:
            home = bl == best[1]
            out["home_found"] = 1
            out["home_area_m2"] = float(home.sum() * mpp * mpp)
            out["home_center_offset_m"] = float(best[0])
    if home is None:  # fall back to rings around the DINS point
        home = np.zeros_like(seg, bool)
        home[int(cy), int(cx)] = True
    d = ndimage.distance_transform_edt(~home) * mpp
    for lo, hi in RINGS:
        ring = (d > lo) & (d <= hi) & ~home
        tot = max(int(ring.sum()), 1)
        for ci, cname in enumerate(CLASSES):
            out[f"r{lo}_{hi}_{cname}"] = float(((seg == ci) & ring).sum() / tot)
        out[f"r{lo}_{hi}_veg"] = out[f"r{lo}_{hi}_tree"] + out[f"r{lo}_{hi}_rangeland"]
    others = (seg == BUILDING) & ~home
    if others.any():
        out["nearest_building_m"] = float(d[others].min())
        lab_o, n_o = ndimage.label(others)
        near_ids = np.unique(lab_o[(d <= 30) & others])
        out["n_buildings_30m"] = int(len(near_ids[near_ids > 0]))
    else:
        out["nearest_building_m"] = float("nan")
        out["n_buildings_30m"] = 0
    for cname in CLASSES:
        out[f"crop_{cname}"] = float((seg == CLASSES.index(cname)).mean())
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = Segmenter().to(dev).eval()
    model.load_state_dict(torch.load(ROOT / "output" / "models" / "oem_segmenter" / "best.pt", map_location=dev))
    recs = collect()
    ids = sorted(recs)
    (OUT / "seg").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(0, len(ids), a.bs):
        batch = ids[i:i + a.bs]
        ims = [np.array(Image.open(recs[g]["imagery"]["image_path"]).convert("RGB")) for g in batch]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).to(dev)
        H, W = x.shape[-2:]
        x = F.pad(x, (0, (-W) % 14, 0, (-H) % 14), mode="reflect")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(norm(x))[:, :, :H, :W].argmax(1).cpu().numpy().astype(np.uint8)
        for g, seg in zip(batch, pred):
            r = recs[g]
            z = r["imagery"].get("zoom", 18)
            mpp = 156543.03392 * math.cos(math.radians(lat_of(r))) / 2 ** z
            Image.fromarray(seg).save(OUT / "seg" / f"{g}.png")
            rows.append({"global_id": g, "event": r["event"], "mpp": mpp,
                         "y": int(r["labels"]["outcome"] == "Destroyed"), **features(seg, mpp)})
        if i % (a.bs * 20) == 0:
            print(i, len(ids), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "features.parquet")
    print(df.shape, "home found", df["home_found"].mean().round(3))


if __name__ == "__main__":
    main()
