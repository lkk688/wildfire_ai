"""
Export a blinded, NON-EXPERT annotation batch (simple visual recognition only).

Expert annotators are not available, so the batch asks only questions a layperson
can answer from the picture (is there a roof at the cross, how many roofs touch
the 50 m circle, are trees over the roof, is there a pool, is the building still
standing in the post-fire image).  No outcome guess and no expert attributes are
asked.  These labels verify VLM perception (extraction answers, positive control),
not fire risk.

Images get a red centre cross and a 50 m circle.  Annotators see no address, no
year built, no fire name.  Each building goes to two annotators (agreement).

Outputs (output/annotation/<batch>/):
  sheet_<annotator>.csv, images/<anon>_pre.jpg / _post.jpg, INSTRUCTIONS.txt
  answer_key.csv  anon_id -> global_id, outcome, inspector attributes (KEEP PRIVATE)

Usage:
  python scripts/data/export_annotation_batch.py --batch simple01 --n 100 --annotators A B C
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
POSTFIRE = ROOT / "output" / "splits" / "pilot_postfire.jsonl"
OUT_ROOT = ROOT / "output" / "annotation"
SEED = 1019
# Non-expert, simple visual questions only (no expert judgement, no outcome guess).
# Each answer is a closed choice; "unsure" is always allowed.
QUESTIONS = {
    "q1_center_is_building": "Is there a building roof at the red-cross centre?  yes / no / unsure",
    "q2_buildings_in_circle": "How many separate building roofs touch the 50 m circle?  0 / 1-2 / 3-5 / 6-10 / >10 / unsure",
    "q3_roof_under_trees": "How much of the centre roof is hidden by tree crowns?  none / <25% / 25-50% / >50% / unsure",
    "q4_trees_touch_roof": "Do any tree crowns touch or overhang the centre roof?  yes / no / unsure",
    "q5_pool": "Is there a swimming pool on the centre lot?  yes / no / unsure",
    "q6_postfire_building_present": "POST-FIRE image: is the centre building still standing?  yes / no / unsure / no image",
}
def overlay(src: str, dst: Path, lat: float) -> None:
    """Red cross at the centre + 50 m circle (Web-Mercator zoom 18 ground resolution)."""
    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    c = im.size[0] // 2
    r = 50.0 / (156543.03392 * math.cos(math.radians(lat)) / 2 ** 18)
    d.ellipse((c - r, c - r, c + r, c + r), outline=(255, 0, 0), width=2)
    d.line((c - 8, c, c + 8, c), fill=(255, 0, 0), width=2)
    d.line((c, c - 8, c, c + 8), fill=(255, 0, 0), width=2)
    im.save(dst, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--annotators", nargs="+", required=True)
    args = ap.parse_args()
    if len(args.annotators) < 2:
        raise SystemExit("need >= 2 annotators for double annotation")

    recs = [json.loads(line) for line in POSTFIRE.read_text().splitlines()]  # diagnostic pilot + post-fire crops
    recs = [r for r in recs if r["imagery"]["accepted"]]
    rng = np.random.default_rng(SEED)
    recs = [recs[i] for i in rng.permutation(len(recs))[: args.n]]

    out = OUT_ROOT / args.batch
    (out / "images").mkdir(parents=True, exist_ok=True)
    key_rows, sheets = [], {a: [] for a in args.annotators}
    for i, r in enumerate(recs):
        anon = hashlib.sha256(f"{SEED}:{args.batch}:{r['global_id']}".encode()).hexdigest()[:10]
        lat = r["inputs"]["prefire"]["latitude"]
        overlay(r["imagery"]["image_path"], out / "images" / f"{anon}_pre.jpg", lat)
        post = r.get("imagery_postfire") or {}
        if post.get("accepted"):
            overlay(post["image_path"], out / "images" / f"{anon}_post.jpg", lat)
        a1 = args.annotators[i % len(args.annotators)]
        a2 = args.annotators[(i + 1) % len(args.annotators)]
        for a in (a1, a2):
            row = {"anon_id": anon, "pre_image": f"images/{anon}_pre.jpg",
                   "post_image": f"images/{anon}_post.jpg" if post.get("accepted") else "none"}
            row.update({f"{k} [{q}]": "" for k, q in QUESTIONS.items()})
            row["notes"] = ""
            sheets[a].append(row)
        key_rows.append({"anon_id": anon, "global_id": r["global_id"], "event": r["event"],
                         "outcome": r["labels"]["outcome"], "damage": r["labels"]["damage"],
                         "annotators": f"{a1};{a2}", "has_postfire": bool(post.get("accepted")),
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
    (out / "INSTRUCTIONS.txt").write_text(
        "Each row is one location. Open pre_image (and post_image if given). The red cross marks the centre; "
        "the red circle has a 50 m radius.\nAnswer each bracketed question with one of the listed options. "
        "Use 'unsure' whenever you cannot tell -- that is a valid, useful answer. Do not look up the address.\n")
    print(f"[OK] {len(recs)} buildings x 2 annotators -> {out}  (answer_key.csv is PRIVATE)")


if __name__ == "__main__":
    main()
