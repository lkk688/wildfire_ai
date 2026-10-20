"""
Sample buildings from a non-test split (train / val) and attach time-verified
pre-fire imagery, for fitting decision heads on VLM-extracted features or image
embeddings (the VLM-as-sensor paradigm).  Same record schema as
output/splits/pilot_test.jsonl.

Per event: up to --k destroyed + --k survived (events need >= --min-per-class of
each), so many events contribute and no mega-fire dominates.

Usage:
  python scripts/data/sample_split_with_imagery.py --split train --k 10
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.build_research_splits import (  # noqa: E402
    IMG_DIR, OUT, SEED, _num, assign_splits, load_residential, to_record)
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = load_residential()
    ev = assign_splits(df)
    df = df.merge(ev[["event", "split"]], on="event")
    df = df[df["split"] == args.split]
    cnt = df.groupby("event")["y"].agg(["sum", "count"])
    ok = cnt[(cnt["sum"] >= args.min_per_class) & (cnt["count"] - cnt["sum"] >= args.min_per_class)].index
    rng = np.random.default_rng(SEED + 7)
    parts = []
    for e, g in df[df["event"].isin(ok)].groupby("event"):
        for y in (0, 1):
            gy = g[g["y"] == y]
            parts.append(gy.sample(n=min(args.k, len(gy)), random_state=int(rng.integers(1 << 31))))
    smp = pd.concat(parts).reset_index(drop=True).assign(pilot_set="fit")
    print(f"[*] {args.split}: {len(smp)} buildings from {smp['event'].nunique()} events")
    releases = load_releases()

    def work(i):
        r = smp.iloc[i]
        yb = _num(r["YEARBUILT"])
        prov = fetch_prefire_crop(float(r["lat"]), float(r["lon"]), r["INCIDENTSTARTDATE_ISO"].date(),
                                  IMG_DIR / f"{r['GLOBALID']}.jpg", releases,
                                  year_built=int(yb) if yb and 1800 <= yb <= 2026 else None)
        return to_record(r, prov.to_dict(), split=args.split)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(work, range(len(smp))))
    out = OUT / f"{args.split}_sample.jsonl"
    with open(out, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    acc = sum(r["imagery"]["accepted"] for r in recs)
    print(f"[OK] {out}: accepted {acc}/{len(recs)};",
          Counter(r["imagery"]["reject_reason"].split(":")[0] for r in recs if not r["imagery"]["accepted"]))


if __name__ == "__main__":
    main()
