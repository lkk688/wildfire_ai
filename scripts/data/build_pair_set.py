"""
Exposure-controlled pair test set: destroyed / survived residential neighbours
in the same held-out (2024+) fire event, within --radius metres.

Neighbours share roughly the same fire exposure (front arrival, wind, ember
load), so "which of these two burned?" isolates the building-level signal that
pre-fire imagery could carry.  Chance = 0.5; the ceiling with clean (fully
observed) inspector attributes is ~0.62-0.65 (output/experiments/exposure_control.md).

Each destroyed home is used at most once, pairs are capped per event, and both
buildings need time-verified imagery.  Records reuse the pilot_test schema.

Output: output/splits/pairs_test.jsonl  ({"pair_id", "event", "dist_m", "destroyed": rec, "survived": rec})
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.build_research_splits import (  # noqa: E402
    IMG_DIR, MIN_PER_CLASS_TEST, OUT, SEED, _num, assign_splits,
    load_residential, to_record)
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402

R_EARTH = 6_371_000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--min-dist", type=float, default=15.0, help="avoid duplicate points / same lot")
    ap.add_argument("--per-event", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS_TEST, help="per event, to be eligible")
    ap.add_argument("--exclude", nargs="*", default=[], help="pair files whose buildings must not be reused")
    ap.add_argument("--out", default="pairs_test.jsonl")
    ap.add_argument("--seed-offset", type=int, default=11)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--near-m", type=float, default=100.0,
                    help="drop buildings within this distance of any record from another split")
    args = ap.parse_args()

    df = load_residential()
    ev = assign_splits(df)
    df = df.merge(ev[["event", "split"]], on="event")
    other = df[df["split"] != args.split]
    mine = df["split"] == args.split
    tree_o = BallTree(np.radians(other[["lat", "lon"]].values), metric="haversine")
    df["dist_other_m"] = np.nan
    df.loc[mine, "dist_other_m"] = tree_o.query(np.radians(df.loc[mine, ["lat", "lon"]].values), k=1)[0][:, 0] * R_EARTH
    elig = ev[(ev["split"] == args.split) & (ev["destroyed"] >= args.min_per_class) &
              (ev["survived"] >= args.min_per_class)]["event"]
    test = df[df["event"].isin(elig) & ~(df["dist_other_m"] < args.near_m)].copy()
    pilot_ids = {json.loads(line)["global_id"] for line in (OUT / "pilot_test.jsonl").read_text().splitlines()}
    for f in args.exclude:
        for line in (OUT / f).read_text().splitlines():
            q = json.loads(line)
            pilot_ids |= {q["destroyed"]["global_id"], q["survived"]["global_id"]}
    test = test[~test["GLOBALID"].isin(pilot_ids)]  # keep pair set disjoint from the pilot / other pair sets

    rng = np.random.default_rng(SEED + args.seed_offset)
    pairs = []
    for e, g in test.groupby("event"):
        g = g.reset_index(drop=True)
        d, s = g[g["y"] == 1], g[g["y"] == 0]
        if not len(d) or not len(s):
            continue
        tree = BallTree(np.radians(s[["lat", "lon"]].values), metric="haversine")
        order = rng.permutation(len(d))
        used_s, n = set(), 0
        for i in order:
            row = d.iloc[i]
            idx, dist = tree.query_radius(np.radians([[row["lat"], row["lon"]]]), args.radius / R_EARTH,
                                          return_distance=True)
            cands = [(j, dd * R_EARTH) for j, dd in zip(idx[0], dist[0])
                     if dd * R_EARTH >= args.min_dist and j not in used_s]
            if not cands:
                continue
            j, dm = cands[int(rng.integers(len(cands)))]
            used_s.add(j)
            pairs.append((e, row, s.iloc[j], dm))
            n += 1
            if n >= args.per_event:
                break
    print(f"[*] {len(pairs)} candidate pairs from {len({p[0] for p in pairs})} events")
    releases = load_releases()

    def crop(r):
        yb = _num(r["YEARBUILT"])
        prov = fetch_prefire_crop(float(r["lat"]), float(r["lon"]), r["INCIDENTSTARTDATE_ISO"].date(),
                                  IMG_DIR / f"{r['GLOBALID']}.jpg", releases,
                                  year_built=int(yb) if yb and 1800 <= yb <= 2026 else None)
        return to_record(r, prov.to_dict(), split=args.split)

    def work(p):
        e, rd, rs, dm = p
        rd, rs = rd.copy(), rs.copy()
        rd["pilot_set"] = rs["pilot_set"] = "pair"
        return {"event": e, "dist_m": round(float(dm), 1), "destroyed": crop(rd), "survived": crop(rs)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(work, pairs))
    ok = [o for o in out if o["destroyed"]["imagery"]["accepted"] and o["survived"]["imagery"]["accepted"]]
    with open(OUT / args.out, "w") as f:
        for k, o in enumerate(ok):
            o["pair_id"] = f"{Path(args.out).stem}_{k:04d}"
            f.write(json.dumps(o) + "\n")
    print(f"[OK] {len(ok)}/{len(out)} pairs with accepted imagery ->", OUT / args.out)
    print(pd.Series([o["event"] for o in ok]).value_counts().to_string())


if __name__ == "__main__":
    main()
