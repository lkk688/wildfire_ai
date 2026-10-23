"""
(Pilot A) Community-scale reasoning from a pre-fire aerial view (output/community/cells.jsonl).

Input: Wayback pre-fire crop at zoom 16 (~1.5 km across) with the community (~500 m cell) outlined, captured before
the fire; told that a wildfire reached the area. Not told the fire's name, place, date, or how many homes were inspected.
Output: probability that at least half of the homes in the box were destroyed, expected loss fraction, and checkable
intermediate facts (distance from homes to continuous wildland, vegetation type, position, density, slope), plus
community-level recommendations.

Scoring (scripts/eval/community_eval.py): AUROC for high loss, rank correlation with the loss fraction, within-fire
pair accuracy, and agreement of the stated distance-to-wildland with ESA WorldCover.
Output: output/community/vlm/<run>/raw.jsonl
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_vlm_conditions import Client  # noqa: E402

OUT = ROOT / "output" / "community"

PROMPT = """You are assessing wildfire structure-loss risk for a residential community.

[Image] Pre-fire aerial photo, north up, about 1.5 km across (~2 m per pixel). The yellow box (~500 m) outlines
the community being assessed. A wildfire later reached this area; assume the community was exposed to it.
[Known before the fire] Median year built of the homes: {yb}.

Judge from the image: where the homes sit relative to continuous wildland vegetation (forest, shrub/chaparral,
grass), how dense and how ordered the housing is, terrain, vegetation inside the community and around homes, roads
and other breaks. Do not try to identify the place or rely on knowledge of real past fires.

Return only JSON:
{{"p_high_loss": <probability 0-1 that at least half of the homes in the box were destroyed>,
 "expected_loss_fraction": <0-1>,
 "dist_to_wildland_m": <estimated typical distance from the homes to continuous wildland vegetation, metres; 0 if the homes are within it>,
 "wildland_type": "forest" | "shrub" | "grass" | "mixed" | "none_nearby",
 "position": "embedded_in_wildland" | "wildland_edge" | "interior_of_development",
 "housing_density": "sparse" | "moderate" | "dense",
 "terrain": "flat" | "rolling" | "steep",
 "key_evidence": [<up to 4 short phrases>],
 "recommendations": [<up to 3 concrete community-level actions>]}}"""


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        j["p_high_loss"] = min(max(float(j["p_high_loss"]), 0.0), 1.0)
        j["expected_loss_fraction"] = min(max(float(j.get("expected_loss_fraction", j["p_high_loss"])), 0.0), 1.0)
        if j.get("dist_to_wildland_m") is not None:
            j["dist_to_wildland_m"] = float(j["dist_to_wildland_m"])
        return j
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--endpoint")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()
    cells = [json.loads(x) for x in (OUT / "cells.jsonl").read_text().splitlines()]
    cells = [c for c in cells if c["split"] in a.splits.split(",") and c["imagery"]["accepted"]]
    if a.limit:
        cells = cells[: a.limit]
    out = OUT / "vlm" / a.run
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = set()
    if raw.exists():
        done = {json.loads(x)["cell_id"] for x in raw.read_text().splitlines() if json.loads(x).get("parsed")}
    todo = [c for c in cells if c["cell_id"] not in done]
    client = Client(a.model, endpoint=a.endpoint, max_tokens=a.max_tokens, no_think=a.no_think)

    def work(c):
        yb = f"{c['yb_median']:.0f}" if c.get("yb_median") == c.get("yb_median") and c.get("yb_median") else "unknown"
        res = client.call(PROMPT.format(yb=yb), Path(c["imagery"]["box_path"]).read_bytes())
        return {"cell_id": c["cell_id"], "event": c["event"], "split": c["split"], "model": a.model, "ok": res.get("ok"),
                "text": res.get("text"), "error": res.get("error"), "parsed": parse(res.get("text"))}

    with raw.open("a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, row in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, len(todo), flush=True)
    print("done", len(todo))


if __name__ == "__main__":
    main()
