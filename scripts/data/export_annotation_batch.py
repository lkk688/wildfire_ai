"""
Export a blinded annotation batch for decisive experiment #1 (human determinability).

Annotators see only pre-fire inputs (time-verified crop + inputs.prefire minus
anything outcome-related) and fill attribute value / visibility status /
confidence and their own P(destroyed).  Outcome, damage, inspector attributes
and event-level damage statistics are withheld; the answer key is written to
a separate file that annotators must not receive.

Each building is assigned to two annotators (double annotation) in a seeded
order so inter-annotator agreement can be computed.

Outputs (output/annotation/<batch>/):
  sheet_<annotator>.csv   one row per building for that annotator
  images/<anon_id>.jpg    crops renamed to anonymous ids (no GLOBALID leak)
  answer_key.csv          anon_id -> global_id, outcome, inspector attributes (KEEP PRIVATE)

Usage:
  python scripts/data/export_annotation_batch.py --batch pilot01 --n 60 --annotators A B C
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "output" / "splits" / "pilot_test.jsonl"
OUT_ROOT = ROOT / "output" / "annotation"
SEED = 1019
ATTRS = ["roof_material", "roof_condition_debris", "vegetation_0_5ft", "vegetation_5_30ft",
         "tree_canopy_overhang", "nearest_structure_m", "deck_or_attached_structure",
         "fence_attached", "outbuildings_within_30ft"]
STATUS = "visible|occluded|insufficient_resolution|time_mismatch_suspected|not_applicable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--pilot-set", default="diagnostic")
    args = ap.parse_args()
    if len(args.annotators) < 2:
        raise SystemExit("need >= 2 annotators for double annotation")

    recs = [json.loads(line) for line in PILOT.read_text().splitlines()]
    recs = [r for r in recs if r["imagery"]["accepted"] and r["pilot_set"] == args.pilot_set]
    rng = np.random.default_rng(SEED)
    recs = [recs[i] for i in rng.permutation(len(recs))[: args.n]]

    out = OUT_ROOT / args.batch
    (out / "images").mkdir(parents=True, exist_ok=True)
    key_rows, sheets = [], {a: [] for a in args.annotators}
    for i, r in enumerate(recs):
        anon = hashlib.sha256(f"{SEED}:{args.batch}:{r['global_id']}".encode()).hexdigest()[:10]
        shutil.copyfile(r["imagery"]["image_path"], out / "images" / f"{anon}.jpg")
        pf = r["inputs"]["prefire"]
        a1 = args.annotators[i % len(args.annotators)]
        a2 = args.annotators[(i + 1) % len(args.annotators)]
        for a in (a1, a2):
            row = {"anon_id": anon, "image": f"images/{anon}.jpg",
                   "structure_type": pf.get("structure_type") or "", "year_built": pf.get("year_built") or "",
                   "county": pf.get("county") or "", "imagery_capture_date": r["imagery"]["capture_date"],
                   "fire_start": pf["incident_start"]}
            for k in ATTRS:
                row[f"{k}__value"] = ""
                row[f"{k}__status({STATUS})"] = ""
                row[f"{k}__confidence(1-5)"] = ""
            row.update({"p_destroyed_0_100": "", "decisive_evidence": "", "minutes_spent": "", "notes": ""})
            sheets[a].append(row)
        key_rows.append({"anon_id": anon, "global_id": r["global_id"], "event": r["event"],
                         "outcome": r["labels"]["outcome"], "damage": r["labels"]["damage"],
                         "annotators": f"{a1};{a2}",
                         **{f"inspector_{k}": v for k, v in r["inputs"]["inspector_postfire"].items()}})

    for a, rows in sheets.items():
        with open(out / f"sheet_{a}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out / "answer_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)
    print(f"[OK] {len(recs)} buildings x 2 annotators -> {out}  (answer_key.csv is PRIVATE)")


if __name__ == "__main__":
    main()
