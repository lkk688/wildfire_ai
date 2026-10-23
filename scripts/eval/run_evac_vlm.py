"""
(Pilot B) VLM evacuation judgement on distance-matched replay items (output/evac/vlm_items.jsonl).

The model sees the map known at decision time t (pre-fire imagery, fire state, community box, wind arrow) and the
facts known at t. It is not told the fire's name, place, date or year. It outputs the probability that the fire will
reach the community and an action tier with reasons. Scored against what happened later.

Evaluation: pair accuracy (reached vs. not-reached community at the same decision point and distance), AUROC,
and false reassurance (reached communities given 'monitor'), with the same items scored by the rule baselines.
Output: output/evac/vlm/<run>/raw.jsonl and summary.json
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_vlm_conditions import Client  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

EVAC = ROOT / "output" / "evac"
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(deg):
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def prompt(it, with_map=True):
    txt = f"""You are advising on wildfire evacuation. Decide whether a residential community should evacuate now.

[Map] Pre-fire aerial imagery. Red area: the fire's mapped extent at the decision time (it may lag the real fire by
hours). Yellow box: the community (~500 m across, about {it['n_buildings']} buildings). Blue arrow: wind direction
(arrow points where the wind blows to). North is up; a scale bar is at the bottom left.

[Known at the decision time]
- Local time of day: {it['local_hour']:02d}:00
- Fire size: {it['area_ha']:.0f} ha; growth over the previous 12 h: {it['growth_ha_h']:.0f} ha/h
- Distance from the fire edge to the community: {it['dist_m'] / 1000:.1f} km; the community lies {compass(it['bearing_from_fire'])} of the nearest fire edge
- Surface wind (reanalysis, may understate local gusts): from {compass(it['wind_from_deg'])} at {it['wind_ms']:.0f} m/s, gusts {it['gust_ms']:.0f} m/s; relative humidity {it['rh']:.0f}%

Reason from the map: fuels and vegetation between the fire and the community, terrain (slopes, canyons, ridges),
barriers (roads, rivers, irrigated or developed land), the wind relative to the community, and the fire's growth.
Do not rely on recognising the place or any real past event.

Return only JSON:
{{"p_reached": <probability 0-1 that the fire will eventually reach this community>,
 "action": "evacuate_now" | "warning_prepare" | "monitor",
 "key_factors": [<up to 4 short phrases>],
 "reasoning": "<2-3 sentences>"}}"""
    if not with_map:  # text-only ablation: same facts, no map
        txt = txt.replace(txt[txt.index("[Map]"):txt.index("[Known at the decision time]")], "")
        txt = txt.replace("Reason from the map: fuels", "No map is available. Reason from the facts: fuels")
        txt = txt.replace("about {n} buildings".format(n=it["n_buildings"]), "")
    return txt


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        p = float(j["p_reached"])
        return {"p_reached": min(max(p, 0.0), 1.0), "action": str(j.get("action")), "key_factors": j.get("key_factors"),
                "reasoning": j.get("reasoning")}
    except Exception:  # noqa: BLE001
        return None


def summarize(rows, items):
    by = {r["item_id"]: r for r in rows}
    p = {k: (v["parsed"]["p_reached"] if v.get("parsed") else 0.5) for k, v in by.items()}  # intention-to-treat
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})[it["role"]] = it
    acc, ev, base = [], [], {"distance": [], "wind_adjusted": []}
    for pid, pr in pairs.items():
        if set(pr) != {"reached", "not_reached"}:
            continue
        a, b = pr["reached"], pr["not_reached"]
        if a["item_id"] not in p or b["item_id"] not in p:
            continue
        acc.append(1.0 if p[a["item_id"]] > p[b["item_id"]] else 0.5 if p[a["item_id"]] == p[b["item_id"]] else 0.0)
        ev.append(a["event"])
        wa = lambda x: x["dist_m"] / (1 + max(0, x["downwind_align"]) * x["wind_ms"] / 5)  # noqa: E731
        base["distance"].append(1.0 if a["dist_m"] < b["dist_m"] else 0.5 if a["dist_m"] == b["dist_m"] else 0.0)
        base["wind_adjusted"].append(1.0 if wa(a) < wa(b) else 0.5 if wa(a) == wa(b) else 0.0)
    acc, ev = np.array(acc), np.array(ev)
    ids = [it["item_id"] for it in items if it["item_id"] in p]
    y = np.array([next(i["reached"] for i in items if i["item_id"] == k) for k in ids])
    s = np.array([p[k] for k in ids])
    act = {k: (by[k]["parsed"] or {}).get("action") if by[k].get("parsed") else None for k in ids}
    pos = [k for k, yy in zip(ids, y) if yy == 1]
    neg = [k for k, yy in zip(ids, y) if yy == 0]
    out = {"n_items": len(ids), "n_pairs": int(len(acc)), "parsed": sum(bool(by[k].get("parsed")) for k in ids),
           "pair_acc": float(acc.mean()), "pair_ci95": _cluster_boot(ev, lambda i: acc[i].mean()),
           "auroc": float(roc_auc_score(y, s)),
           "baseline_pair_acc": {k: float(np.mean(v)) for k, v in base.items()},
           "false_reassurance_monitor": float(np.mean([act[k] == "monitor" for k in pos])),
           "evacuate_rate_not_reached": float(np.mean([act[k] == "evacuate_now" for k in neg])),
           "evacuate_rate_reached": float(np.mean([act[k] == "evacuate_now" for k in pos])),
           "per_event_pair_acc": {e: float(acc[ev == e].mean()) for e in sorted(set(ev))}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--endpoint")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--no-map", action="store_true", help="text-only ablation")
    a = ap.parse_args()
    items = [json.loads(x) for x in (EVAC / "vlm_items.jsonl").read_text().splitlines()]
    uniq = list({it["item_id"]: it for it in items}.values())
    if a.limit:
        keep = {it["pair_id"] for it in items[: a.limit]}
        items = [it for it in items if it["pair_id"] in keep]
        uniq = list({it["item_id"]: it for it in items}.values())
    out = EVAC / "vlm" / a.run
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = {}
    if raw.exists():
        for x in raw.read_text().splitlines():
            r = json.loads(x)
            if r.get("parsed") or r["item_id"] not in done:
                done[r["item_id"]] = r
    todo = [it for it in uniq if not (done.get(it["item_id"]) or {}).get("parsed")]
    client = Client(a.model, endpoint=a.endpoint, max_tokens=a.max_tokens, no_think=a.no_think)

    def work(it):
        res = client.call(prompt(it, not a.no_map), None if a.no_map else Path(it["map_path"]).read_bytes())
        row = {"item_id": it["item_id"], "event": it["event"], "model": a.model, "ok": res.get("ok"),
               "text": res.get("text"), "error": res.get("error"), "parsed": parse(res.get("text"))}
        return row

    with raw.open("a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, row in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[row["item_id"]] = row
            if i % 25 == 0:
                print(i, len(todo), flush=True)
    s = summarize(list(done.values()), items)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
