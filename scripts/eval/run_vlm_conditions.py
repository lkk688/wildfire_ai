"""
Decisive experiments #2 (visual increment) and #3 (prompt bias), Review 1019.

The legacy evaluator (src/vlm_benchmark/evaluate_baselines.py) sent only text --
DINS post-fire attributes plus the same "Red Flag, 25-35 mph, DFM<6%" sentence
for every case -- so its numbers are a text-model experiment.  This harness runs
the same model under controlled conditions on the held-out pilot test set
(output/splits/pilot_test.jsonl) and actually attaches the pre-fire image.

Evidence conditions (what the model receives):
  meta            pre-fire metadata only (structure type, year built, event, county)
  image           metadata + time-verified pre-fire crop of this building
  image_mismatch  metadata + crop of a *different* building from the same event
                  (fixed seeded derangement).  If image ~= image_mismatch, the
                  model is not using the pixels.
  inspector       metadata + DINS post-fire inspector attributes (legacy-style
                  text input; outcome-correlated missingness -> leaky oracle)
Context conditions (independent of evidence):
  neutral         no weather sentence
  extreme         the legacy "Red Flag ... DFM < 6%" sentence, identical for all
Every call is logged with the prompt, image SHA256, the model id the API
*reports*, raw response, token usage, latency and parse status, so numbers can
be re-derived without re-querying.  Runs are resumable.

Usage:
  python scripts/eval/run_vlm_conditions.py --model claude-haiku-5-5 \
      --evidence meta image image_mismatch --context neutral extreme \
      --pilot-set diagnostic --limit 0 --workers 4
  python scripts/eval/run_vlm_conditions.py --model claude-haiku-5-5 --report-only
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import summarize_run  # noqa: E402

PILOT = ROOT / "output" / "splits" / "pilot_test.jsonl"
OUT_ROOT = ROOT / "output" / "experiments" / "vlm_conditions"
PROMPT_VERSION = "v1-2026-10-19"
SEED = 1019

EXTREME_WEATHER = ("Weather & Fuel: Red Flag Warning conditions, wind gusts 25-35 mph, dead fuel moisture "
                   "(DFM) < 6%, surrounding chaparral/timber brush.")
ATTR_KEYS = ["roof", "vegetation_within_5ft", "vegetation_5_30ft", "tree_canopy_overhang",
             "nearest_structure_distance", "deck_or_attached_structure", "fence_attached"]

INSPECTOR_LABELS = {
    "roofconstruction": "Roof", "eaves": "Eaves", "ventscreen": "Vent screen",
    "exteriorsiding": "Exterior siding", "windowpane": "Window pane",
    "deckporchongrade": "Deck/porch on grade", "deckporchelevated": "Deck/porch elevated",
    "patiocovercarport": "Patio cover/carport", "fenceattachedtostructure": "Fence attached",
}


def build_prompt(rec: dict, evidence: str, context: str) -> str:
    pf = rec["inputs"]["prefire"]
    lines = [
        "You are assessing wildfire vulnerability of one residential building in a California "
        "wildland-urban interface fire. The fire did reach this community.",
        "",
        "[Pre-fire record]",
        f"- Structure type: {pf.get('structure_type') or 'not recorded'}",
        f"- Year built: {pf.get('year_built') or 'not recorded'}",
        f"- Fire event: {pf.get('incident_name')} ({pf.get('county')} County), started {pf.get('incident_start')}",
    ]
    if evidence in ("image", "image_mismatch"):
        lines += ["", "[Imagery]",
                  "- Attached: a pre-fire aerial orthophoto (~0.3-0.6 m/pixel, 512x512 px, about 150 m across). "
                  "The building being assessed is at the exact centre of the image."]
    if evidence == "postfire_image":
        days = -int(rec["imagery_postfire"]["staleness_days"])
        lines += ["", "[Imagery]",
                  f"- Attached: a POST-fire aerial orthophoto captured {days} days after the fire started "
                  "(~0.3-0.6 m/pixel, 512x512 px, about 150 m across). The building being assessed is at the "
                  "exact centre of the image. (Positive control: judge the building's state as shown.)"]
    if evidence == "prepost_image":
        days = -int(rec["imagery_postfire"]["staleness_days"])
        lines += ["", "[Imagery]",
                  "- Attached: two aerial orthophotos of the same place (~0.3-0.6 m/pixel, about 150 m across). "
                  f"Image 1 is PRE-fire (captured {rec['imagery']['capture_date']}); image 2 is POST-fire "
                  f"(captured {days} days after the fire started). The building being assessed is at the exact "
                  "centre of both. Compare them."]
    if evidence == "inspector":
        lines += ["", "[Building attributes]"]
        for k, lab in INSPECTOR_LABELS.items():
            lines.append(f"- {lab}: {rec['inputs']['inspector_postfire'].get(k) or 'Unknown'}")
    if context == "extreme":
        lines += ["", "[Fire environment]", f"- {EXTREME_WEATHER}"]
    lines += [
        "",
        "[Task]",
        "Estimate the probability that this building was destroyed (>25% damage) in this fire. "
        "Base the estimate only on the evidence provided; say when an attribute cannot be determined.",
        "",
        "Reply with a single JSON object and nothing else:",
        "{",
        '  "p_destroyed": <integer 0-100>,',
        '  "prediction": "Destroyed" or "Survived",',
        '  "attributes": {' + ", ".join(f'"{k}": {{"value": <string>, "status": "visible|occluded|'
                                      f'insufficient_resolution|no_image"}}' for k in ATTR_KEYS[:2]) + ", ...},",
        f'  "attribute_keys_required": {json.dumps(ATTR_KEYS)},',
        '  "key_evidence": [<up to 3 short strings>]',
        "}",
        "Fill \"attributes\" for every key in attribute_keys_required (omit attribute_keys_required itself).",
    ]
    return "\n".join(lines)


def derangement_map(recs: list) -> dict:
    """global_id -> global_id of another building in the same event (seeded, no fixed points)."""
    rng = np.random.default_rng(SEED)
    out = {}
    by_ev = {}
    for r in recs:
        by_ev.setdefault(r["event"], []).append(r["global_id"])
    for ids in by_ev.values():
        if len(ids) < 2:
            continue
        perm = list(ids)
        while True:
            rng.shuffle(perm)
            if all(a != b for a, b in zip(ids, perm)):
                break
        out.update(dict(zip(ids, perm)))
    return out


class Client:
    def __init__(self, model: str, endpoint: str = None, api_key: str = None, max_tokens: int = 1200):
        self.model = model
        self.max_tokens = max_tokens
        self.is_claude = model.lower().startswith("claude")
        if self.is_claude:
            self.url = endpoint or "https://api.anthropic.com/v1/messages"
            self.key = api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        elif "qwen" in model.lower():
            base = endpoint or os.environ.get("TB_REMOTE_QWEN_URL", "http://REMOTE_HOST:28000/v1")
            self.url = base if base.endswith("/chat/completions") else base.rstrip("/") + "/chat/completions"
            self.key = api_key or os.environ.get("QWEN_REMOTE_KEY", "")
        else:
            self.url = endpoint or os.environ.get("LLM_ENDPOINT", "https://api.minimaxi.com/v1/chat/completions")
            self.key = api_key or os.environ.get("MINIMAX_API_KEY", "")

    def call(self, prompt: str, image_bytes=None) -> dict:
        """image_bytes: None, one JPEG (bytes) or a list of JPEGs (sent in order, before the text)."""
        t0 = time.perf_counter()
        images = [] if image_bytes is None else ([image_bytes] if isinstance(image_bytes, bytes) else list(image_bytes))
        if self.is_claude:
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(b).decode()}} for b in images]
            content.append({"type": "text", "text": prompt})
            # Current Claude models reject `temperature`; sampling uses the API default.
            payload = {"model": self.model, "max_tokens": self.max_tokens,
                       "messages": [{"role": "user", "content": content}]}
            headers = {"Content-Type": "application/json", "x-api-key": self.key,
                       "anthropic-version": "2023-06-01"}
        else:
            if images:
                content = [{"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}} for b in images]
                content.append({"type": "text", "text": prompt})
            else:
                content = prompt
            payload = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": content}]}
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code not in (429, 500, 502, 503, 529):
                    return {"ok": False, "error": last, "latency": round(time.perf_counter() - t0, 2)}
            except Exception as e:  # noqa: BLE001
                last = str(e)
            time.sleep(5 * (attempt + 1))
        else:
            return {"ok": False, "error": last, "latency": round(time.perf_counter() - t0, 2)}
        lat = round(time.perf_counter() - t0, 2)
        if self.is_claude:
            text = "\n".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
            return {"ok": True, "text": text, "reported_model": j.get("model"), "usage": j.get("usage"),
                    "stop_reason": j.get("stop_reason"), "latency": lat}
        msg = (j.get("choices") or [{}])[0].get("message", {})
        return {"ok": True, "text": msg.get("content") or "", "reasoning": msg.get("reasoning") or
                msg.get("reasoning_content"), "reported_model": j.get("model"), "usage": j.get("usage"),
                "stop_reason": (j.get("choices") or [{}])[0].get("finish_reason"), "latency": lat}


def parse_response(text: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return {"parse_ok": False, "parse_error": "no_json"}
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"parse_ok": False, "parse_error": f"json:{e}"[:120]}
    p = j.get("p_destroyed")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return {"parse_ok": False, "parse_error": "no_p_destroyed"}
    if not 0 <= p <= 100:
        return {"parse_ok": False, "parse_error": "p_out_of_range"}
    pred = str(j.get("prediction", "")).strip().title()
    if pred not in ("Destroyed", "Survived"):
        pred = "Destroyed" if p >= 50 else "Survived"
    return {"parse_ok": True, "p_destroyed": p / 100.0, "prediction": pred,
            "attributes": j.get("attributes"), "key_evidence": j.get("key_evidence")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint")
    ap.add_argument("--evidence", nargs="+", default=["meta", "image", "image_mismatch"],
                    choices=["meta", "image", "image_mismatch", "inspector", "postfire_image", "prepost_image"])
    ap.add_argument("--context", nargs="+", default=["neutral", "extreme"], choices=["neutral", "extreme"])
    ap.add_argument("--pilot-set", default="diagnostic", choices=["diagnostic", "natural", "all"])
    ap.add_argument("--limit", type=int, default=0, help="max buildings (0 = all), taken balanced per event")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1200, help="raise if replies hit max_tokens")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--records", default=str(PILOT))
    args = ap.parse_args()

    run = args.run_name or re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out_dir = OUT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    if not args.report_only:
        recs = [json.loads(line) for line in Path(args.records).read_text().splitlines()]
        if {"postfire_image", "prepost_image"} & set(args.evidence):
            recs = [r for r in recs if r.get("imagery_postfire", {}).get("accepted")]
        recs = [r for r in recs if r["imagery"]["accepted"]
                and (args.pilot_set == "all" or r["pilot_set"] == args.pilot_set)]
        if args.limit:
            # round-robin over (event, outcome) strata so a small limit stays balanced
            strata = {}
            for r in sorted(recs, key=lambda r: r["global_id"]):
                strata.setdefault((r["event"], r["labels"]["outcome"]), []).append(r)
            pick, i = [], 0
            while len(pick) < args.limit and any(i < len(v) for v in strata.values()):
                for k in sorted(strata):
                    if i < len(strata[k]) and len(pick) < args.limit:
                        pick.append(strata[k][i])
                i += 1
            recs = pick
        by_id = {r["global_id"]: r for r in recs}
        swap = derangement_map(recs)

        done = set()
        if raw_path.exists():
            for line in raw_path.read_text().splitlines():
                j = json.loads(line)
                if j.get("parse_ok"):  # truncated / unparsable replies are retried
                    done.add((j["global_id"], j["evidence"], j["context"]))

        jobs = [(r, e, c) for r in recs for e in args.evidence for c in args.context
                if (r["global_id"], e, c) not in done and not (e == "image_mismatch" and r["global_id"] not in swap)]
        print(f"[*] {args.model}: {len(recs)} buildings, {len(jobs)} calls to run ({len(done)} cached)")
        client = Client(args.model, args.endpoint, max_tokens=args.max_tokens)
        lock = threading.Lock()

        def work(job):
            r, e, c = job
            prompt = build_prompt(r, e, c)
            img_rec = None
            if e == "image":
                img_rec = r
            elif e == "image_mismatch":
                img_rec = by_id[swap[r["global_id"]]]
            img = Path(img_rec["imagery"]["image_path"]).read_bytes() if img_rec else None
            if e == "postfire_image":
                img_rec = r
                img = Path(r["imagery_postfire"]["image_path"]).read_bytes()
            if e == "prepost_image":
                img_rec = r
                img = [Path(r["imagery"]["image_path"]).read_bytes(), Path(r["imagery_postfire"]["image_path"]).read_bytes()]
            res = client.call(prompt, img)
            row = {
                "global_id": r["global_id"], "event": r["event"], "pilot_set": r["pilot_set"],
                "evidence": e, "context": c, "label": r["labels"]["outcome"],
                "requested_model": args.model, "reported_model": res.get("reported_model"),
                "endpoint": client.url, "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt,
                "image_global_id": img_rec["global_id"] if img_rec else None,
                "image_sha256": (hashlib.sha256(img).hexdigest() if isinstance(img, bytes)
                                 else [hashlib.sha256(b).hexdigest() for b in img]) if img else None,
                "response_ok": res["ok"], "error": res.get("error"), "raw_text": res.get("text"),
                "reasoning": res.get("reasoning"), "usage": res.get("usage"),
                "stop_reason": res.get("stop_reason"), "latency_sec": res.get("latency"),
                "max_tokens": args.max_tokens,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            row.update(parse_response(res.get("text", "")) if res["ok"] else {"parse_ok": False,
                                                                                "parse_error": "request_failed"})
            with lock:
                with open(raw_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            return row["parse_ok"]

        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, ok in enumerate(ex.map(work, jobs), 1):
                n_ok += bool(ok)
                if i % 25 == 0 or i == len(jobs):
                    print(f"    {i}/{len(jobs)} done, parsed {n_ok}")

    summary = summarize_run(raw_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(summary["markdown"])
    print(summary["markdown"])


if __name__ == "__main__":
    main()
