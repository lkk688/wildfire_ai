"""
Confirmatory direct-judgment set from the untouched VAL events (2019-2023).

No VLM has been run on any val-event building before this set was built
(see docs/CONFIRMATORY_PLAN.md).  Balanced per event: up to --k destroyed +
--k survived from events with >= --min-per-class of each, excluding buildings
within 100 m of another split and buildings already in the confirmatory pair set.

Output: output/splits/confirm_direct.jsonl (pilot_test schema, pilot_set="diagnostic")
"""

import argparse
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
from scripts.data.build_research_splits import (  # noqa: E402
    IMG_DIR, OUT, SEED, _num, assign_splits, load_residential, to_record)
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402

R_EARTH = 6_371_000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min-per-class", type=int, default=7)
    ap.add_argument("--exclude", default="confirm_pairs.jsonl")
    args = ap.parse_args()

    df = load_residential()
    ev = assign_splits(df)
    df = df.merge(ev[["event", "split"]], on="event")
    other = df[df["split"] != "val"]
    val = df[df["split"] == "val"].copy()
    tree = BallTree(np.radians(other[["lat", "lon"]].values), metric="haversine")
    val["dist_other_m"] = tree.query(np.radians(val[["lat", "lon"]].values), k=1)[0][:, 0] * R_EARTH
    val = val[val["dist_other_m"] >= 100]
    excl = set()
    if (OUT / args.exclude).exists():
        for line in (OUT / args.exclude).read_text().splitlines():
            q = json.loads(line)
            excl |= {q["destroyed"]["global_id"], q["survived"]["global_id"]}
    val = val[~val["GLOBALID"].isin(excl)]
    cnt = val.groupby("event")["y"].agg(["sum", "count"])
    ok = cnt[(cnt["sum"] >= args.min_per_class) & (cnt["count"] - cnt["sum"] >= args.min_per_class)].index
    rng = np.random.default_rng(SEED + 31)
    parts = []
    for e, g in val[val["event"].isin(ok)].groupby("event"):
        for y in (0, 1):
            gy = g[g["y"] == y]
            parts.append(gy.sample(n=min(args.k, len(gy)), random_state=int(rng.integers(1 << 31))))
    smp = pd.concat(parts).reset_index(drop=True).assign(pilot_set="diagnostic")
    print(f"[*] confirm_direct: {len(smp)} buildings from {smp['event'].nunique()} val events")
    rel = load_releases()

    def work(i):
        r = smp.iloc[i]
        yb = _num(r["YEARBUILT"])
        prov = fetch_prefire_crop(float(r["lat"]), float(r["lon"]), r["INCIDENTSTARTDATE_ISO"].date(),
                                  IMG_DIR / f"{r['GLOBALID']}.jpg", rel,
                                  year_built=int(yb) if yb and 1800 <= yb <= 2026 else None)
        return to_record(r, prov.to_dict(), split="val")

    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(work, range(len(smp))))
    with open(OUT / "confirm_direct.jsonl", "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    acc = [r for r in recs if r["imagery"]["accepted"]]
    print(f"[OK] accepted {len(acc)}/{len(recs)};",
          Counter(r["imagery"]["reject_reason"].split(":")[0] for r in recs if not r["imagery"]["accepted"]))
    print(Counter((r["event"], r["labels"]["outcome"]) for r in acc))


if __name__ == "__main__":
    main()
