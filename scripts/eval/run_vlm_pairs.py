"""
Exposure-controlled forced choice: which of two neighbouring homes was destroyed?

Each item in output/splits/pairs_test.jsonl is a destroyed and a survived home
in the same held-out fire, 15-100 m apart.  The model sees both pre-fire crops
(order seeded per pair) and is told exactly one was destroyed, so the
base-rate / "everything burns" bias cannot help, and shared exposure (wind,
front arrival) is roughly controlled.  Score = pair accuracy (0.5 chance);
reference ceiling with clean inspector attributes ~0.62-0.65.

Evidence: image (both crops) | meta (structure type + year built only)

Usage:
  python scripts/eval/run_vlm_pairs.py --model claude-sonnet-5-5 --evidence image meta
"""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_vlm_conditions import Client  # noqa: E402

PAIRS = ROOT / "output" / "splits" / "pairs_test.jsonl"
OUT_ROOT = ROOT / "output" / "experiments" / "vlm_pairs"
PROMPT_VERSION = "pairs-v1-2026-10-19"
SEED = 1019


def _desc(rec: dict) -> str:
    pf = rec["inputs"]["prefire"]
    return f"{pf.get('structure_type') or 'type not recorded'}, built {pf.get('year_built') or 'year not recorded'}"


def build_prompt(pair: dict, a: dict, b: dict, evidence: str) -> str:
    pf = a["inputs"]["prefire"]
    lines = [f"Two residential buildings, A and B, stand {pair['dist_m']:.0f} m apart in the same neighbourhood. "
             f"The {pf['incident_name']} fire ({pf['county']} County, started {pf['incident_start']}) reached both. "
             "Exactly one of them was destroyed; the other survived.", ""]
    if evidence == "image":
        lines.append("Attached are two pre-fire aerial orthophotos (about 0.3-0.6 m/pixel, roughly 150 m across). "
                     "The first image is centred on building A, the second on building B.")
    lines += [f"- Building A: {_desc(a)}", f"- Building B: {_desc(b)}", "",
              "Which one was more likely destroyed? Base your answer only on the evidence given.",
              'Reply with a single JSON object and nothing else: {"p_A_destroyed": <integer 0-100>, '
              '"destroyed": "A" or "B", "key_differences": [<up to 3 short strings>]}']
    return "\n".join(lines)


def parse(text: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return {"parse_ok": False, "parse_error": "no_json"}
    try:
        j = json.loads(m.group(0))
        p = float(j["p_A_destroyed"]) / 100.0
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return {"parse_ok": False, "parse_error": str(e)[:120]}
    if not 0 <= p <= 1:
        return {"parse_ok": False, "parse_error": "p_out_of_range"}
    return {"parse_ok": True, "p_A_destroyed": p, "key_differences": j.get("key_differences")}


def summarize(raw_path: Path) -> str:
    best = {}
    for line in raw_path.read_text().splitlines():
        r = json.loads(line)
        k = (r["pair_id"], r["evidence"])
        if k not in best or r.get("parse_ok"):
            best[k] = r
    rng = np.random.default_rng(SEED)
    L = [f"# Pair forced choice: `{raw_path.parent.name}`\n",
         "| Evidence | Parsed/Req | Pair accuracy [95% CI] | Mean p(correct) | Picked A |", "|---|---:|---|---:|---:|"]
    for ev in sorted({k[1] for k in best}):
        rs = [r for k, r in best.items() if k[1] == ev]
        ok = [r for r in rs if r.get("parse_ok")]
        if not ok:
            continue
        pc = np.array([r["p_A_destroyed"] if r["destroyed_is"] == "A" else 1 - r["p_A_destroyed"] for r in ok])
        acc = (pc > 0.5) + 0.5 * (pc == 0.5)
        evs = np.array([r["event"] for r in ok])
        uniq = np.unique(evs)
        boots = [acc[np.concatenate([np.where(evs == e)[0] for e in rng.choice(uniq, len(uniq))])].mean()
                 for _ in range(1000)]
        pick_a = np.mean([r["p_A_destroyed"] > 0.5 for r in ok])
        L.append(f"| {ev} | {len(ok)}/{len(rs)} | {acc.mean():.3f} [{np.percentile(boots, 2.5):.3f}, "
                 f"{np.percentile(boots, 97.5):.3f}] | {pc.mean():.3f} | {pick_a:.0%} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--evidence", nargs="+", default=["image", "meta"], choices=["image", "meta"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--run-name")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--pairs", nargs="+", default=[str(PAIRS)])
    args = ap.parse_args()
    run = args.run_name or re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out_dir = OUT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    if not args.report_only:
        pairs = [json.loads(line) for f in args.pairs for line in Path(f).read_text().splitlines()]
        done = set()
        if raw_path.exists():
            done = {(j["pair_id"], j["evidence"]) for j in map(json.loads, raw_path.read_text().splitlines())
                    if j.get("parse_ok")}
        jobs = [(p, e) for p in pairs for e in args.evidence if (p["pair_id"], e) not in done]
        print(f"[*] {args.model}: {len(pairs)} pairs, {len(jobs)} calls ({len(done)} cached)")
        client = Client(args.model, max_tokens=args.max_tokens)
        lock = threading.Lock()

        def work(job):
            p, e = job
            flip = int(hashlib.sha256(f"{SEED}:{p['pair_id']}".encode()).hexdigest(), 16) % 2 == 1
            a, b = (p["survived"], p["destroyed"]) if flip else (p["destroyed"], p["survived"])
            prompt = build_prompt(p, a, b, e)
            imgs = [Path(x["imagery"]["image_path"]).read_bytes() for x in (a, b)] if e == "image" else None
            res = client.call(prompt, imgs)
            row = {"pair_id": p["pair_id"], "event": p["event"], "dist_m": p["dist_m"], "evidence": e,
                   "destroyed_is": "B" if flip else "A", "requested_model": args.model,
                   "reported_model": res.get("reported_model"), "prompt_version": PROMPT_VERSION,
                   "prompt": prompt, "image_sha256": [hashlib.sha256(x).hexdigest() for x in imgs] if imgs else None,
                   "response_ok": res["ok"], "error": res.get("error"), "raw_text": res.get("text"),
                   "usage": res.get("usage"), "stop_reason": res.get("stop_reason"),
                   "latency_sec": res.get("latency"), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            row.update(parse(res.get("text", "")) if res["ok"] else {"parse_ok": False, "parse_error": "request_failed"})
            with lock:
                with open(raw_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            return row["parse_ok"]

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            n = sum(bool(x) for x in ex.map(work, jobs))
        print(f"    parsed {n}/{len(jobs)}")

    md = summarize(raw_path)
    (out_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
