"""
VLM-as-sensor: perception-only attribute extraction from pre-fire aerial crops.

The model is never told about a fire or asked for an outcome.  It fills a fixed
schema of aerial-observable attributes, each a closed category (with an explicit
"cannot_determine") plus a 1-5 confidence.  Closed categories make every answer
(a) checkable against an independent verifier where one exists (DINS roof /
deck records for standing homes, DINS-point geometry for spacing, rotation
invariance), and (b) usable as features for a decision head fitted on training
events -- instead of trusting the LLM's own loss prior.

Variants: orig (as captured) and rot90 (image rotated 90 deg; answers should
not change -> self-consistency check without any labels).

Usage:
  python scripts/eval/run_vlm_extraction.py --model Qwen3.8-27B \
      --records output/splits/pilot_test.jsonl output/splits/train_sample.jsonl --variants orig
"""

import argparse
import hashlib
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_vlm_conditions import Client  # noqa: E402

OUT_ROOT = ROOT / "output" / "experiments" / "vlm_extraction"
PROMPT_VERSION = "extract-v1-2026-10-19"
CD = "cannot_determine"

SCHEMA = {
    "center_building_identifiable": ["yes", "partially", "no"],
    "roof_material": ["asphalt_shingle", "clay_concrete_tile", "metal", "flat_membrane_or_gravel",
                      "wood_shake", "other", CD],
    "roof_under_tree_canopy": ["none", "lt25pct", "25_50pct", "gt50pct", CD],
    "tree_canopy_within_30m": ["none_lt10pct", "low_10_30pct", "moderate_30_60pct", "high_gt60pct", CD],
    "shrubs_or_plants_within_1.5m_of_walls": ["none", "some", "dense", CD],
    "unmanaged_wildland_vegetation_within_30m": ["yes", "no", CD],
    "unmanaged_wildland_vegetation_within_100m": ["yes", "no", CD],
    "nearest_other_building": ["lt5m", "5_10m", "10_20m", "20_50m", "gt50m", CD],
    "buildings_within_50m": ["0", "1_2", "3_5", "6_10", "gt10", CD],
    "outbuilding_or_shed_within_10m": ["yes", "no", CD],
    "attached_deck_or_patio_cover": ["yes", "no", CD],
    "swimming_pool_on_lot": ["yes", "no", CD],
    "setting": ["dense_suburban", "suburban_large_lots", "rural_residential", "isolated_wildland"],
    "building_footprint_size": ["small_lt150m2", "medium_150_300m2", "large_gt300m2", CD],
    "image_quality": ["good", "fair", "poor"],
}


def build_prompt() -> str:
    lines = [
        "You are annotating a pre-fire aerial orthophoto (about 0.3-0.6 m per pixel, 512x512 px, roughly 150 m "
        "across, north up). The building to annotate is the one at the exact centre of the image.",
        "Report only what is visible. If an attribute cannot be judged at this resolution, because of occlusion "
        f"or image quality, answer \"{CD}\" rather than guessing.",
        "",
        "Return a single JSON object and nothing else. For every key, give {\"v\": <one allowed value>, "
        "\"c\": <confidence 1-5>}. Keys and allowed values:",
    ]
    for k, vals in SCHEMA.items():
        lines.append(f'- "{k}": {" | ".join(vals)}')
    lines.append('Also add "notes": <one short sentence on what limits your view, or "">.')
    return "\n".join(lines)


def parse(text: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return {"parse_ok": False, "parse_error": "no_json"}
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"parse_ok": False, "parse_error": f"json:{e}"[:120]}
    out, bad = {}, []
    for k, vals in SCHEMA.items():
        x = j.get(k)
        v = x.get("v") if isinstance(x, dict) else x
        c = x.get("c") if isinstance(x, dict) else None
        v = str(v).strip() if v is not None else None
        if v not in vals:
            bad.append(k)
            v = None
        try:
            c = int(c)
        except (TypeError, ValueError):
            c = None
        out[k] = {"v": v, "c": c}
    return {"parse_ok": len(bad) <= 2, "invalid_keys": bad, "attrs": out, "notes": j.get("notes")}


def load_image(path: str, variant: str) -> bytes:
    raw = Path(path).read_bytes()
    if variant == "orig":
        return raw
    im = Image.open(io.BytesIO(raw)).rotate(90, expand=False)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def iter_records(paths):
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if "destroyed" in r and "survived" in r:  # pair file
                yield r["destroyed"]
                yield r["survived"]
            elif r["imagery"]["accepted"]:
                yield r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["orig"], choices=["orig", "rot90"])
    ap.add_argument("--limit", type=int, default=0, help="per records file order; 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--run-name")
    args = ap.parse_args()

    run = args.run_name or re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out_dir = OUT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            j = json.loads(line)
            if j.get("parse_ok"):
                done.add((j["global_id"], j["variant"]))

    recs, seen = [], set()
    for r in iter_records(args.records):
        if r["global_id"] not in seen:
            seen.add(r["global_id"])
            recs.append(r)
    if args.limit:
        recs = recs[: args.limit]
    jobs = [(r, v) for r in recs for v in args.variants if (r["global_id"], v) not in done]
    print(f"[*] {args.model}: {len(recs)} buildings, {len(jobs)} calls ({len(done)} cached)")
    client = Client(args.model, max_tokens=args.max_tokens)
    prompt = build_prompt()
    lock = threading.Lock()

    def work(job):
        r, v = job
        img = load_image(r["imagery"]["image_path"], v)
        res = client.call(prompt, img)
        row = {"global_id": r["global_id"], "event": r["event"], "split": r["split"], "variant": v,
               "requested_model": args.model, "reported_model": res.get("reported_model"),
               "prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "image_sha256": hashlib.sha256(img).hexdigest(), "response_ok": res["ok"],
               "error": res.get("error"), "raw_text": res.get("text"), "usage": res.get("usage"),
               "stop_reason": res.get("stop_reason"), "latency_sec": res.get("latency"),
               "max_tokens": args.max_tokens, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        row.update(parse(res.get("text", "")) if res["ok"] else {"parse_ok": False, "parse_error": "request_failed"})
        with lock:
            with open(raw_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row["parse_ok"]

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, ok in enumerate(ex.map(work, jobs), 1):
            n_ok += bool(ok)
            if i % 50 == 0 or i == len(jobs):
                print(f"    {i}/{len(jobs)} done, parsed {n_ok}", flush=True)


if __name__ == "__main__":
    main()
