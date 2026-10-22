"""
(C1) T1-XL: ~1,000 test-fire (2024+) residential buildings sampled at each event's NATURAL prevalence
(proportional to event size, capped per event), disjoint from the pilot set and all pair sets,
with time-verified pre-fire imagery.  Output: output/splits/test_natural_xl.jsonl
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
from scripts.data.build_research_splits import IMG_DIR, OUT, SEED, _num, assign_splits, load_residential, to_record  # noqa: E402
from src.vlm_benchmark.imagery_provenance import fetch_prefire_crop, load_releases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=250)
    args = ap.parse_args()
    df = load_residential()
    ev = assign_splits(df)
    df = df.merge(ev[["event", "split"]], on="event")
    test = df[df["split"] == "test"]
    used = {json.loads(l)["global_id"] for l in (OUT / "pilot_test.jsonl").read_text().splitlines()}
    for f in OUT.glob("pairs_test*.jsonl"):
        for l in f.read_text().splitlines():
            q = json.loads(l)
            used |= {q["destroyed"]["global_id"], q["survived"]["global_id"]}
    test = test[~test["GLOBALID"].isin(used)]
    sizes = test.groupby("event").size()
    alloc = np.minimum((sizes / sizes.sum() * args.n * 1.15).round().astype(int), args.cap)
    rng = np.random.default_rng(SEED + 101)
    smp = pd.concat([g.sample(n=min(int(alloc[e]), len(g)), random_state=int(rng.integers(1 << 31)))
                     for e, g in test.groupby("event") if alloc[e] > 0]).assign(pilot_set="natural_xl")
    print(f"[*] sampled {len(smp)} from {smp['event'].nunique()} events; destroyed rate {smp['y'].mean():.2f}", flush=True)
    rel = load_releases()

    def work(i):
        r = smp.iloc[i]
        yb = _num(r["YEARBUILT"])
        prov = fetch_prefire_crop(float(r["lat"]), float(r["lon"]), r["INCIDENTSTARTDATE_ISO"].date(),
                                  IMG_DIR / f"{r['GLOBALID']}.jpg", rel, year_built=int(yb) if yb and 1800 <= yb <= 2026 else None)
        return to_record(r, prov.to_dict(), split="test")

    with ThreadPoolExecutor(max_workers=10) as ex:
        recs = list(ex.map(work, range(len(smp))))
    with open(OUT / "test_natural_xl.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = [r for r in recs if r["imagery"]["accepted"]]
    print(f"[OK] accepted {len(acc)}/{len(recs)}", Counter(r["imagery"]["reject_reason"].split(":")[0] for r in recs if not r["imagery"]["accepted"]))


if __name__ == "__main__":
    main()
