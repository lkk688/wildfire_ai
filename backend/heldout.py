"""
Held-out test cases for the VLM view: stored per-case outputs of this project's evaluation
(output/demo/heldout_cases.json, written by scripts/demo/export_heldout_cases.py) and an on-demand re-run of a
case with the exact research prompt (scripts/eval/run_vlm_conditions.py / run_vlm_pairs.py).
"""
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "output" / "demo" / "heldout_cases.json"
SPL = ROOT / "output" / "splits"

_lock = threading.Lock()
_cache: Dict[str, Any] = {"mtime": None, "data": None, "recs": None}

LIVE_MODELS = {
    "claude-sonnet-5-5": {"label": "Claude Sonnet 5.5", "env": ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")},
    "claude-haiku-5-5": {"label": "Claude Haiku 5.5", "env": ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")},
    "Qwen3.8-27B": {"label": "Qwen3.8-27B", "env": ("TB_REMOTE_QWEN_URL",)},
}


def _load() -> Dict[str, Any]:
    if not EXPORT.exists():
        raise FileNotFoundError(f"{EXPORT.relative_to(ROOT)} not found; run scripts/demo/export_heldout_cases.py")
    m = EXPORT.stat().st_mtime
    with _lock:
        if _cache["mtime"] != m:
            _cache["data"] = json.loads(EXPORT.read_text())
            _cache["mtime"] = m
            _cache["recs"] = None
        return _cache["data"]


def _records() -> Dict[str, Any]:
    """Full split records (needed to rebuild the exact research prompt for a live re-run)."""
    _load()
    with _lock:
        if _cache["recs"] is None:
            t3 = {}
            for line in (SPL / "postfire_test.jsonl").read_text().splitlines():
                r = json.loads(line)
                t3[r["global_id"]] = r
            t2 = {}
            for f in ("pairs_test.jsonl", "pairs_test_large.jsonl"):
                for line in (SPL / f).read_text().splitlines():
                    q = json.loads(line)
                    t2[q["pair_id"]] = q
            _cache["recs"] = {"t3": t3, "t2": t2}
        return _cache["recs"]


def summary() -> Dict[str, Any]:
    d = _load()
    events = {}
    for task in ("t3", "t2"):
        ev = {}
        for c in d[task]["cases"]:
            ev[c["event"]] = ev.get(c["event"], 0) + 1
        events[task] = sorted(ev.items(), key=lambda kv: -kv[1])
    return {
        "source": f"{EXPORT.relative_to(ROOT)} (scripts/demo/export_heldout_cases.py)",
        "note": d.get("note"),
        "t3": {"models": d["t3"]["models"], "n": len(d["t3"]["cases"]), "split_file": d["t3"]["split_file"],
               "events": events["t3"]},
        "t2": {"models": d["t2"]["models"], "n": len(d["t2"]["cases"]), "split_file": d["t2"]["split_file"],
               "events": events["t2"]},
        "live_models": [{"id": k, "label": v["label"], "configured": any(os.environ.get(e) for e in v["env"])}
                        for k, v in LIVE_MODELS.items()],
    }


def cases(task: str) -> Dict[str, Any]:
    d = _load()
    if task not in ("t3", "t2"):
        raise KeyError(task)
    return {"task": task, "cases": d[task]["cases"]}


def image_path(global_id: str, epoch: str) -> Optional[Path]:
    d = _load()
    p = d["images"].get(global_id, {}).get(epoch)
    return Path(p) if p and Path(p).exists() else None


def live_predict(task: str, case_id: str, model: str, condition: str) -> Dict[str, Any]:
    """Re-run one held-out case now with the research prompt and the same verified images."""
    if model not in LIVE_MODELS:
        raise ValueError(f"unsupported model {model}")
    from scripts.eval import run_vlm_conditions as rc
    recs = _records()
    client = rc.Client(model, max_tokens=12000 if "qwen" in model.lower() else 3000)
    if task == "t3":
        r = recs["t3"].get(case_id)
        if r is None:
            raise KeyError(case_id)
        if condition not in ("postfire_image", "prepost_image"):
            raise ValueError("condition must be postfire_image or prepost_image")
        prompt = rc.build_prompt(r, condition, "neutral")
        post = Path(r["imagery_postfire"]["image_path"]).read_bytes()
        imgs = post if condition == "postfire_image" else [Path(r["imagery"]["image_path"]).read_bytes(), post]
        res = client.call(prompt, imgs)
        parsed = rc.parse_response(res.get("text", "")) if res.get("ok") else {"parse_ok": False}
        p = parsed.get("p_destroyed") if parsed.get("parse_ok") else None
        return {"success": bool(res.get("ok")), "error": res.get("error"), "model_id": res.get("reported_model"),
                "latency_sec": res.get("latency"), "prompt_version": rc.PROMPT_VERSION, "prompt": prompt,
                "parsed": p is not None, "p": p if p is not None else 0.5,
                "evidence": parsed.get("key_evidence") or [], "raw_text": (res.get("text") or "")[:4000]}
    if task == "t2":
        from scripts.eval import run_vlm_pairs as rp
        q = recs["t2"].get(case_id)
        if q is None:
            raise KeyError(case_id)
        flip = int(hashlib.sha256(f"{rp.SEED}:{q['pair_id']}".encode()).hexdigest(), 16) % 2 == 1  # same A/B as the stored run
        a, b = (q["survived"], q["destroyed"]) if flip else (q["destroyed"], q["survived"])
        ev = "meta" if condition == "meta" else "image"
        prompt = rp.build_prompt(q, a, b, ev)
        imgs = [Path(x["imagery"]["image_path"]).read_bytes() for x in (a, b)] if ev == "image" else None
        res = client.call(prompt, imgs)
        parsed = rp.parse(res.get("text", "")) if res.get("ok") else {"parse_ok": False}
        pa = parsed.get("p_A_destroyed") if parsed.get("parse_ok") else None
        p_true = None if pa is None else (pa if not flip else 1 - pa)
        return {"success": bool(res.get("ok")), "error": res.get("error"), "model_id": res.get("reported_model"),
                "latency_sec": res.get("latency"), "prompt_version": rp.PROMPT_VERSION, "prompt": prompt,
                "destroyed_is": "B" if flip else "A", "parsed": p_true is not None,
                "p_destroyed_one": p_true if p_true is not None else 0.5,
                "differences": parsed.get("key_differences") or [], "raw_text": (res.get("text") or "")[:4000]}
    raise ValueError("task must be t3 or t2")
