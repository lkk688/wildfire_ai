"""
Export the held-out test cases and the stored per-case model outputs for the web demo (VLM view).

Nothing here is computed for display only: every probability is read from a stored experiment output, every label
from CAL FIRE DINS, every image path and capture date from the verified Wayback records in output/splits/.

  T3  post-fire recognition: output/splits/postfire_test.jsonl (2024-25 fires, both crops verified)
      - zero-shot VLMs: output/experiments/vlm_conditions/pf_test_*/raw.jsonl (post-fire image, pre+post images)
      - Qwen3.5 LoRA adapters: output/experiments/postfire/sft_Qwen3.5-{4B,9B}/test_probs.npy
        (token readout p(major)+p(destroyed); rows follow sft_postfire_qwen.load("test"))
      - frozen DINOv2-base linear probes: output/experiments/postfire/lp_test_preds.json
  T2  pre-fire same-fire neighbour pairs: output/splits/pairs_test.jsonl + pairs_test_large.jsonl (403 main pairs)
      - output/experiments/vlm_pairs/pairs_*/raw.jsonl (image and metadata-only conditions)

Unparsed answers are kept as p = 0.5 (intention-to-treat), as in the evaluation scripts.
Privacy: street addresses, APNs and assessed values are not exported; coordinates are rounded to 3 decimals
(~100 m) for the map link only.

  python3 scripts/demo/export_heldout_cases.py
Output: output/demo/heldout_cases.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SPL = ROOT / "output" / "splits"
EXP = ROOT / "output" / "experiments"
OUT = ROOT / "output" / "demo" / "heldout_cases.json"

# key -> (label, run dir, evidence, kind)
T3_VLM = [
    ("sonnet_post", "Claude Sonnet 5.5", "pf_test_claude_sonnet_5_5", "postfire_image"),
    ("sonnet_prepost", "Claude Sonnet 5.5", "pf_test_claude_sonnet_5_5", "prepost_image"),
    ("haiku_post", "Claude Haiku 5.5", "pf_test_claude_haiku_5_5", "postfire_image"),
    ("haiku_prepost", "Claude Haiku 5.5", "pf_test_claude_haiku_5_5", "prepost_image"),
    ("qwen38_post", "Qwen3.8-27B", "pf_test_qwen38", "postfire_image"),
    ("qwen38_prepost", "Qwen3.8-27B", "pf_test_qwen38", "prepost_image"),
    ("qwen35_4b_post", "Qwen3.5-4B (zero-shot)", "pf_test_qwen35_4b_nothink", "postfire_image"),
]
T3_TRAINED = [
    ("lora_4b", "Qwen3.5-4B + LoRA (trained on 2015-23 fires)", "sft_Qwen3.5-4B"),
    ("lora_9b", "Qwen3.5-9B + LoRA (trained on 2015-23 fires)", "sft_Qwen3.5-9B"),
]
T3_LP = [
    ("dinov2_post", "DINOv2-base linear probe", "lp_post facebook/dinov2-base", "postfire_image"),
    ("dinov2_prepost", "DINOv2-base linear probe", "lp_prepost facebook/dinov2-base", "prepost_image"),
]
EVIDENCE_TEXT = {"postfire_image": "post-fire image", "prepost_image": "pre + post images", "image": "two pre-fire images",
                 "meta": "metadata only (no image)"}
T2_RUNS = [
    ("sonnet_image", "Claude Sonnet 5.5", "pairs_sonnet", "image"),
    ("sonnet_meta", "Claude Sonnet 5.5", "pairs_sonnet", "meta"),
    ("haiku_image", "Claude Haiku 5.5", "pairs_haiku", "image"),
    ("qwen38_image", "Qwen3.8-27B", "pairs_qwen38", "image"),
    ("minimax_image", "MiniMax-M3", "pairs_minimax_m3", "image"),
]


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def strlist(x, n=3):
    if isinstance(x, list):
        return [str(s)[:240] for s in x[:n]]
    return []


def building(r, with_post=False):
    pf = r["inputs"]["prefire"]
    im = r["imagery"]
    out = {
        "global_id": r["global_id"],
        "structure_type": pf.get("structure_type"),
        "year_built": pf.get("year_built"),
        "dins_damage": r["labels"]["damage"],
        "label": "Destroyed" if r["labels"]["outcome"] == "Destroyed" else "Survived",
        "lat3": round(float(pf["latitude"]), 3) if pf.get("latitude") is not None else None,
        "lon3": round(float(pf["longitude"]), 3) if pf.get("longitude") is not None else None,
        "pre": {"capture_date": im.get("capture_date"), "provider": im.get("capture_provider"),
                "days_before_fire": im.get("staleness_days"), "release": im.get("release_identifier")},
    }
    if with_post:
        po = r["imagery_postfire"]
        out["post"] = {"capture_date": po.get("capture_date"), "provider": po.get("capture_provider"),
                       "days_after_fire": -int(po["staleness_days"]) if po.get("staleness_days") is not None else None,
                       "release": po.get("release_identifier")}
    return out


def fire(r):
    pf = r["inputs"]["prefire"]
    return {"event": r["event"], "fire": pf.get("incident_name"), "county": pf.get("county"),
            "fire_start": pf.get("incident_start")}


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None


def export_t3(images):
    recs = [r for r in jl(SPL / "postfire_test.jsonl") if r["imagery"]["accepted"] and r["imagery_postfire"]["accepted"]]
    cases = {r["global_id"]: {**fire(r), **building(r, with_post=True), "preds": {}} for r in recs}
    for r in recs:
        images[r["global_id"]] = {"pre": r["imagery"]["image_path"], "post": r["imagery_postfire"]["image_path"]}
    models = []
    for key, label, run, ev in T3_VLM:
        f = EXP / "vlm_conditions" / run / "raw.jsonl"
        if not f.exists():
            continue
        for j in jl(f):
            if j.get("evidence") != ev or j.get("context") != "neutral" or j["global_id"] not in cases:
                continue
            ok = str(j.get("parse_ok")) == "True"
            p = fnum(j.get("p_destroyed")) if ok else None
            prev = cases[j["global_id"]]["preds"].get(key)
            if prev and (prev["parsed"] or p is None):  # retries: keep the parsed answer
                continue
            cases[j["global_id"]]["preds"][key] = {
                "p": p if p is not None else 0.5, "parsed": p is not None,
                "evidence": strlist(j.get("key_evidence")), "ts": j.get("ts"), "model_id": j.get("reported_model")}
        models.append({"key": key, "label": label, "input": EVIDENCE_TEXT[ev], "kind": "zero-shot VLM",
                       "source": str(f.relative_to(ROOT))})
    order = [r["global_id"] for r in recs]
    for key, label, run in T3_TRAINED:
        f = EXP / "postfire" / run / "test_probs.npy"
        if not f.exists():
            continue
        P = np.load(f)
        assert len(P) == len(order), f"{f}: {len(P)} rows vs {len(order)} test records"
        for g, row in zip(order, P):
            cases[g]["preds"][key] = {"p": float(row[3:].sum()), "parsed": True}
        models.append({"key": key, "label": label, "input": "post-fire image", "kind": "fine-tuned VLM",
                       "source": str(f.relative_to(ROOT))})
    lpf = EXP / "postfire" / "lp_test_preds.json"
    if lpf.exists():
        lp = json.loads(lpf.read_text())
        for key, label, col, ev in T3_LP:
            if col not in lp:
                continue
            for g, p in zip(lp["global_id"], lp[col]):
                if g in cases:
                    cases[g]["preds"][key] = {"p": float(p), "parsed": True}
            models.append({"key": key, "label": label, "input": EVIDENCE_TEXT[ev], "kind": "frozen encoder + linear probe",
                           "source": str(lpf.relative_to(ROOT))})
    for m in models:
        cs = [c for c in cases.values() if m["key"] in c["preds"]]
        m["n"] = len(cs)
        y = [c["label"] == "Destroyed" for c in cs]
        p = [c["preds"][m["key"]]["p"] for c in cs]
        m["auroc"] = auroc(y, p)
        m["accuracy_at_0_5"] = float(np.mean([(pp >= 0.5) == yy for pp, yy in zip(p, y)])) if cs else None
        m["n_unparsed"] = sum(not c["preds"][m["key"]]["parsed"] for c in cs)
    return {"cases": list(cases.values()), "models": models, "split_file": "output/splits/postfire_test.jsonl"}


def export_t2(images):
    pairs = jl(SPL / "pairs_test.jsonl") + jl(SPL / "pairs_test_large.jsonl")
    out = {}
    for q in pairs:
        d, s = q["destroyed"], q["survived"]
        for r in (d, s):
            images[r["global_id"]] = {"pre": r["imagery"]["image_path"]}
        # A/B order shown to the models (scripts/eval/run_vlm_pairs.py, seed 1019); the demo shows the same order
        flip = int(hashlib.sha256(f"1019:{q['pair_id']}".encode()).hexdigest(), 16) % 2 == 1
        out[q["pair_id"]] = {"pair_id": q["pair_id"], **fire(d), "dist_m": round(float(q["dist_m"]), 1),
                             "destroyed_is": "B" if flip else "A",
                             "destroyed": building(d), "survived": building(s), "preds": {}}
    models = []
    for key, label, run, ev in T2_RUNS:
        f = EXP / "vlm_pairs" / run / "raw.jsonl"
        if not f.exists():
            continue
        for j in jl(f):
            if j.get("evidence") != ev or j["pair_id"] not in out:
                continue
            ok = str(j.get("parse_ok")) == "True" and fnum(j.get("p_A_destroyed")) is not None
            pa = fnum(j.get("p_A_destroyed")) if ok else 0.5
            # probability the model gave to the building that was actually destroyed
            assert j["destroyed_is"] == out[j["pair_id"]]["destroyed_is"], j["pair_id"]
            p_true = pa if j["destroyed_is"] == "A" else 1 - pa
            prev = out[j["pair_id"]]["preds"].get(key)
            if prev and (prev["parsed"] or not ok):  # retries: keep the parsed answer
                continue
            out[j["pair_id"]]["preds"][key] = {"p_destroyed_one": p_true, "parsed": ok,
                                               "differences": strlist(j.get("key_differences")), "ts": j.get("ts"),
                                               "model_id": j.get("reported_model")}
        models.append({"key": key, "label": label, "input": EVIDENCE_TEXT[ev], "kind": "zero-shot VLM",
                       "source": str(f.relative_to(ROOT))})
    for m in models:
        m["n"] = sum(m["key"] in c["preds"] for c in out.values())
        ps = [c["preds"][m["key"]]["p_destroyed_one"] for c in out.values() if m["key"] in c["preds"]]
        m["pair_accuracy"] = float(np.mean([(p > 0.5) + 0.5 * (p == 0.5) for p in ps])) if ps else None
        m["n_unparsed"] = sum(not c["preds"][m["key"]]["parsed"] for c in out.values() if m["key"] in c["preds"])
    return {"cases": list(out.values()), "models": models,
            "split_file": "output/splits/pairs_test.jsonl + pairs_test_large.jsonl"}


def main():
    images = {}
    res = {"t3": export_t3(images), "t2": export_t2(images), "images": images,
           "note": "Per-case stored outputs of the held-out evaluation; unparsed answers count as p = 0.5."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res))
    for t in ("t3", "t2"):
        print(t, len(res[t]["cases"]), "cases")
        for m in res[t]["models"]:
            print("  ", m["key"], m["n"], {k: round(v, 3) for k, v in m.items() if k in ("auroc", "pair_accuracy") and v})


if __name__ == "__main__":
    main()
