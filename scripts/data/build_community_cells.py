"""
(Pilot A) Community-scale cells for neighbourhood-level reasoning.

A cell is a 0.005-degree grid square (~500 m) inside one fire with >= MIN_HOMES inspected residential DINS records.
Label: fraction of inspected homes destroyed (Destroyed + Major), and high_loss = fraction >= 0.5.
Image: Wayback pre-fire crop centred on the cell centre at zoom 16 (~2 m/px, 768 px ~ 1.5 km), with verified
capture date before the fire (same provenance rules as the building-level benchmark). A copy with the cell outlined
is written for the VLM.

Output: output/community/cells.jsonl, output/community/img/<cell_id>.jpg, <cell_id>_box.jpg
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.build_research_splits import load_residential  # noqa: E402
from scripts.eval.tabular_baselines_on_pilot import SPLITS  # noqa: E402
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402

OUT = ROOT / "output" / "community"
G = 0.005
MIN_HOMES = 10
ZOOM, SIZE = 16, 768


def mpp(lat, z=ZOOM):
    return 156543.03392 * np.cos(np.radians(lat)) / 2 ** z


def build_cells():
    d = load_residential().merge(pd.read_parquet(SPLITS)[["GLOBALID", "split"]], on="GLOBALID")
    d["iy"], d["ix"] = (d.lat // G).astype(int), (d.lon // G).astype(int)
    d["log_value"] = np.log1p(pd.to_numeric(d["ASSESSEDIMPROVEDVALUE"], errors="coerce"))
    d["yb"] = pd.to_numeric(d["YEARBUILT"], errors="coerce").where(lambda x: x.between(1850, 2026))
    d["mobile"] = d["STRUCTURETYPE"].astype(str).str.contains("Mobile", case=False)
    agg = d.groupby(["event", "iy", "ix"]).agg(
        split=("split", "first"), fire_start=("INCIDENTSTARTDATE_ISO", "first"), n=("y", "size"),
        n_destroyed=("y", "sum"), yb_median=("yb", "median"), yb_missing=("yb", lambda x: float(x.isna().mean())),
        log_value_median=("log_value", "median"), mobile_frac=("mobile", "mean"),
        lat_mean=("lat", "mean"), lon_mean=("lon", "mean")).reset_index()
    agg = agg[(agg["n"] >= MIN_HOMES) & agg["split"].isin(["train", "val", "test"])].copy()
    agg["frac_destroyed"] = agg["n_destroyed"] / agg["n"]
    agg["high_loss"] = (agg["frac_destroyed"] >= 0.5).astype(int)
    agg["lat_c"] = (agg["iy"] + 0.5) * G
    agg["lon_c"] = (agg["ix"] + 0.5) * G
    agg["cell_id"] = (agg["event"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True) + "_" + agg["iy"].astype(str)
                      + "_" + agg["ix"].astype(str))
    return agg


def draw_box(src, dst, lat):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    m = mpp(lat)
    hx = G / 2 * 111320 * np.cos(np.radians(lat)) / m
    hy = G / 2 * 110574 / m
    dr = ImageDraw.Draw(im)
    dr.rectangle([w / 2 - hx, h / 2 - hy, w / 2 + hx, h / 2 + hy], outline=(255, 255, 0), width=3)
    im.save(dst, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=300)
    ap.add_argument("--splits", default="test,val,train")
    a = ap.parse_args()
    cells = build_cells()
    rng = np.random.default_rng(1019)
    parts = []
    for sp in a.splits.split(","):
        c = cells[cells["split"] == sp]
        if sp == "train" and len(c) > a.max_train:
            c = c.iloc[rng.choice(len(c), a.max_train, replace=False)]
        parts.append(c)
    cells = pd.concat(parts)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    releases = load_releases()
    done = {}
    out = OUT / "cells.jsonl"
    if out.exists():
        done = {json.loads(x)["cell_id"]: x for x in out.read_text().splitlines()}
    with out.open("a") as f:
        for i, r in enumerate(cells.itertuples()):
            if r.cell_id in done:
                continue
            p = OUT / "img" / f"{r.cell_id}.jpg"
            start = pd.Timestamp(r.fire_start).date()
            prov = fetch_prefire_crop(r.lat_c, r.lon_c, start, p, releases, zoom=ZOOM, size=SIZE)
            pv = prov.to_dict()
            if prov.accepted:
                draw_box(p, OUT / "img" / f"{r.cell_id}_box.jpg", r.lat_c)
            rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in r._asdict().items() if k != "Index"}
            rec["fire_start"] = str(start)
            rec["imagery"] = {**pv, "image_path": str(p), "box_path": str(OUT / "img" / f"{r.cell_id}_box.jpg")}
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, r.split, r.cell_id, prov.accepted, prov.reject_reason, flush=True)


if __name__ == "__main__":
    main()
