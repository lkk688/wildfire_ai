"""
Post-fire damage-recognition dataset (direction 2): DINS outcome from aerial
imagery captured AFTER the fire, with the same event-level splits as the risk
work (train = 2019-2023 train events, val = val events, test = 2024+ events;
dev events excluded).

Per record: a post-fire crop (earliest Wayback capture after the fire start,
<= --max-lag-days) and the time-verified pre-fire crop, so both "post only" and
"pre + post" (change detection) conditions can be run.  Labels: binary
(Destroyed+Major vs No Damage/Affected/Minor) and the 5-level DINS damage.

Sampling per event and class, capped so mega-fires do not dominate:
up to --k-train / --k-eval per (event, damage level).  Coverage (how many
sampled buildings HAVE a post-fire capture) is reported per event, because
post-fire collections exist mainly for large fires -> selection bias to report.

Outputs: output/splits/postfire_{train,val,test}.jsonl, postfire_summary.md
Images:  /data/wildfire_benchmark_data/wayback_crops_postfire/, .../wayback_crops/
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.build_research_splits import (  # noqa: E402
    IMG_DIR, OUT, SEED, _num, assign_splits, load_residential, to_record)
from src.vlm_benchmark.imagery_provenance import (  # noqa: E402
    fetch_postfire_crop_fast, fetch_prefire_crop, load_releases)

POST_DIR = Path("/data/wildfire_benchmark_data/wayback_crops_postfire")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-train", type=int, default=40, help="per (event, damage level) in train")
    ap.add_argument("--k-eval", type=int, default=40, help="per (event, damage level) in val/test")
    ap.add_argument("--max-lag-days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    args = ap.parse_args()

    df = load_residential()
    ev = assign_splits(df)
    df = df.merge(ev[["event", "split"]], on="event")
    rng = np.random.default_rng(SEED + 53)
    rel = load_releases()
    lines = ["# Post-fire damage-recognition dataset\n"]

    for split in args.splits:
        k = args.k_train if split == "train" else args.k_eval
        g = df[df["split"] == split]
        parts = []
        for (e, dmg), gg in g.groupby(["event", "DAMAGE"]):
            parts.append(gg.sample(n=min(k, len(gg)), random_state=int(rng.integers(1 << 31))))
        smp = pd.concat(parts).reset_index(drop=True).assign(pilot_set="postfire")
        print(f"[*] {split}: {len(smp)} sampled from {smp['event'].nunique()} events", flush=True)

        def work(i):
            r = smp.iloc[i]
            start = r["INCIDENTSTARTDATE_ISO"].date()
            post = fetch_postfire_crop_fast(float(r["lat"]), float(r["lon"]), start,
                                            POST_DIR / f"{r['GLOBALID']}.jpg", rel, max_lag_days=args.max_lag_days)
            if not post.accepted:
                return None, post.reject_reason, r["event"]
            yb = _num(r["YEARBUILT"])
            pre = fetch_prefire_crop(float(r["lat"]), float(r["lon"]), start, IMG_DIR / f"{r['GLOBALID']}.jpg", rel,
                                     year_built=int(yb) if yb and 1800 <= yb <= 2026 else None)
            rec = to_record(r, pre.to_dict(), split=split)
            rec["imagery_postfire"] = post.to_dict()
            return rec, None, r["event"]

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(work, range(len(smp))))
        recs = [x for x, _, _ in res if x is not None]
        with open(OUT / f"postfire_{split}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        cov = defaultdict(lambda: [0, 0])
        for x, _, e in res:
            cov[e][1] += 1
            cov[e][0] += x is not None
        lab = Counter(r["labels"]["damage"] for r in recs)
        lines += [f"## {split}: {len(recs)}/{len(smp)} with post-fire capture <= {args.max_lag_days} d",
                  f"Damage levels: {dict(lab)}; pre-fire crop accepted: "
                  f"{sum(r['imagery']['accepted'] for r in recs)}/{len(recs)}\n",
                  "| Event | With post-fire / sampled |", "|---|---:|"]
        lines += [f"| {e.replace(' | ', ' (')}) | {a}/{b} |" for e, (a, b) in sorted(cov.items(), key=lambda x: -x[1][1])]
        lines.append("")
        print(f"[OK] {split}: {len(recs)}/{len(smp)}", Counter(r for _, r, _ in res if r), flush=True)
        (OUT / "postfire_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
