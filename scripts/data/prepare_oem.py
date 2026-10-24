"""
Prepare OpenEarthMap for scripts/train/train_oem_segmenter.py: convert GeoTIFF images/labels to PNG and collect
the official train/val lists. Images whose RGB is not distributed (the xBD-derived regions) are skipped.

  python3 scripts/data/prepare_oem.py --src /data/wildfire_benchmark_data/openearthmap/extracted
Output: /data/wildfire_benchmark_data/openearthmap/png/{images,labels}/<name>.png and splits.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

DST = Path("/data/wildfire_benchmark_data/openearthmap/png")


def read_list(p: Path):
    return [l.strip().replace(".tif", "") for l in p.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = Path(a.src)
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)
    lists = {}
    for kind in ("train", "val", "test"):
        cands = sorted(src.rglob(f"{kind}.txt"))
        if cands:
            lists[kind] = read_list(cands[0])
    imgs = {p.stem: p for p in src.rglob("images/*.tif")}
    labs = {p.stem: p for p in src.rglob("labels/*.tif")}
    ok, skipped = set(), 0
    for name, ip in imgs.items():
        lp = labs.get(name)
        if lp is None:
            continue
        try:
            with rasterio.open(ip) as s:
                im = s.read([1, 2, 3]) if s.count >= 3 else None
            with rasterio.open(lp) as s:
                lb = s.read(1)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if im is None:
            skipped += 1
            continue
        Image.fromarray(np.transpose(im, (1, 2, 0)).astype(np.uint8)).save(DST / "images" / f"{name}.png")
        Image.fromarray(lb.astype(np.uint8)).save(DST / "labels" / f"{name}.png")
        ok.add(name)
    split = {k: [n for n in v if n in ok] for k, v in lists.items()}
    (DST / "splits.json").write_text(json.dumps(split))
    print({k: len(v) for k, v in split.items()}, "converted", len(ok), "skipped", skipped,
          "lists found", {k: len(v) for k, v in lists.items()})


if __name__ == "__main__":
    main()
