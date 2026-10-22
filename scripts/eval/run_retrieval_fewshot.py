"""
Retrieval-augmented baselines for post-fire damage recognition (docs/POSTFIRE_TASK_PLAN.md section 10).

Memory: train+val post-fire crops (DINOv2-base embeddings, cached), cosine retrieval.
  knn        P(destroyed) = similarity-weighted mean label of top-k neighbours
  fewshot    VLM sees k=4 labelled exemplar crops then the query crop
             condition "retrieved": the 4 nearest neighbours; "random": 2+2 seeded random exemplars
Query set: the fixed 400-building test subset used for the Claude runs (plus full test for kNN).

Usage:
  python scripts/eval/run_retrieval_fewshot.py --knn-only
  QWEN_REMOTE_KEY=... python scripts/eval/run_retrieval_fewshot.py --model Qwen3.8-27B --conditions retrieved random
  python scripts/eval/run_retrieval_fewshot.py --model claude-sonnet-5-5 --conditions retrieved
  python scripts/eval/run_retrieval_fewshot.py --report
Output: output/experiments/postfire/retrieval/
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_vlm_conditions import Client, parse_response  # noqa: E402
from scripts.train.sft_postfire_qwen import load  # noqa: E402

PF = ROOT / "output" / "experiments" / "postfire"
OUT = PF / "retrieval"
EXP = ROOT / "output" / "experiments"
SEED = 1019
K_SHOT = 4
LV = {"No Damage": "no damage", "Affected (>0-10%)": "affected (1-10%)", "Minor (10-25%)": "minor (10-25%)",
      "Major (25-50%)": "major (25-50%)", "Destroyed (>50%)": "destroyed (>50%)"}


def emb(split):
    z = np.load(PF / f"emb_dinov2base_{split}.npz", allow_pickle=True)
    d = z["E"].shape[1] // 2
    E = z["E"][:, :d]
    return E / np.linalg.norm(E, axis=1, keepdims=True), list(z["ids"])


def setup():
    mem = load("train") + load("val")
    Em = np.concatenate([emb("train")[0], emb("val")[0]])
    te = load("test")
    Et, ids = emb("test")
    assert ids == [r["global_id"] for r in te]
    ym = np.array([r["labels"]["outcome"] == "Destroyed" for r in mem], dtype=int)
    sims = Et @ Em.T
    return mem, Em, ym, te, sims


def subset_ids():
    best = {}
    for line in (EXP / "vlm_conditions" / "pf_test_claude_sonnet_5_5" / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] == "postfire_image":
            best[r["global_id"]] = 1
    return set(best)


def prompt_text(exemplars):
    lines = ["You are assessing residential buildings after California wildfires from POST-fire aerial "
             "orthophotos (about 0.3-0.6 m/pixel, 512x512 px, about 150 m across). In every image the building "
             "of interest is at the exact centre.", "",
             f"Images 1-{len(exemplars)} are reference examples from OTHER fires, with their CAL FIRE inspection result:"]
    for i, r in enumerate(exemplars, 1):
        lines.append(f"- Image {i}: {LV[r['labels']['damage']]} -> {'DESTROYED' if r['labels']['outcome'] == 'Destroyed' else 'SURVIVED'}")
    lines += ["", f"Image {len(exemplars) + 1} is the building to assess. Estimate the probability that it was destroyed "
              "(>25% damage). Use the references only as visual guidance; judge the query image itself.", "",
              'Reply with a single JSON object and nothing else: {"p_destroyed": <integer 0-100>, '
              '"prediction": "Destroyed" or "Survived", "key_evidence": [<up to 3 short strings>]}']
    return "\n".join(lines)


def run_vlm(args, mem, ym, te, sims):
    sub = subset_ids()
    q = [i for i, r in enumerate(te) if r["global_id"] in sub]
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(ym == 1)[0], np.where(ym == 0)[0]
    out_dir = OUT / re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"
    done = set()
    if raw.exists():
        done = {(j["global_id"], j["condition"]) for j in map(json.loads, raw.read_text().splitlines()) if j.get("parse_ok")}
    jobs = []
    for i in q:
        for c in args.conditions:
            if (te[i]["global_id"], c) in done:
                continue
            if c == "retrieved":
                ex = list(np.argsort(-sims[i])[:K_SHOT])
            else:
                r2 = np.random.default_rng(int(hashlib.sha256(te[i]["global_id"].encode()).hexdigest()[:8], 16) ^ SEED)
                ex = list(r2.choice(pos, 2, replace=False)) + list(r2.choice(neg, 2, replace=False))
                r2.shuffle(ex)
            jobs.append((i, c, ex))
    print(f"[*] {args.model}: {len(q)} queries, {len(jobs)} calls ({len(done)} cached)", flush=True)
    client = Client(args.model, max_tokens=args.max_tokens)
    lock = threading.Lock()

    def work(job):
        i, c, ex = job
        exr = [mem[j] for j in ex]
        prompt = prompt_text(exr)
        imgs = [Path(r["imagery_postfire"]["image_path"]).read_bytes() for r in exr] + \
               [Path(te[i]["imagery_postfire"]["image_path"]).read_bytes()]
        res = client.call(prompt, imgs)
        row = {"global_id": te[i]["global_id"], "event": te[i]["event"], "condition": c,
               "label": te[i]["labels"]["outcome"], "exemplar_ids": [r["global_id"] for r in exr],
               "exemplar_labels": [int(ym[j]) for j in ex], "exemplar_sims": [float(sims[i, j]) for j in ex],
               "requested_model": args.model, "reported_model": res.get("reported_model"),
               "prompt": prompt, "response_ok": res["ok"], "error": res.get("error"), "raw_text": res.get("text"),
               "usage": res.get("usage"), "stop_reason": res.get("stop_reason"), "latency_sec": res.get("latency"),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        row.update(parse_response(res.get("text", "")) if res["ok"] else {"parse_ok": False, "parse_error": "request_failed"})
        with lock:
            with open(raw, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row["parse_ok"]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        n = sum(bool(x) for x in ex.map(work, jobs))
    print(f"    parsed {n}/{len(jobs)}")


def report(mem, ym, te, sims):
    sub = subset_ids()
    yt = np.array([r["labels"]["outcome"] == "Destroyed" for r in te], dtype=int)
    ev = np.array([r["event"] for r in te])
    gid = [r["global_id"] for r in te]
    scores = {}
    for k in (4, 10, 25, 50):
        top = np.argsort(-sims, axis=1)[:, :k]
        w = np.take_along_axis(sims, top, axis=1)
        scores[f"kNN-{k} (embedding)"] = dict(zip(gid, (w * ym[top]).sum(1) / w.sum(1)))
    top4 = np.argsort(-sims, axis=1)[:, :4]
    scores["kNN-4 unweighted vote"] = dict(zip(gid, ym[top4].mean(1)))
    for d in sorted(OUT.glob("*/raw.jsonl")):
        best = {}
        for line in d.read_text().splitlines():
            r = json.loads(line)
            k = (r["global_id"], r["condition"])
            if k not in best or r.get("parse_ok"):
                best[k] = r
        for c in sorted({k[1] for k in best}):
            scores[f"{d.parent.name} few-shot {c}"] = {g: r["p_destroyed"] for (g, cc), r in best.items()
                                                       if cc == c and r.get("parse_ok")}
    for name, run in [("qwen3_8_27b zero-shot", "pf_test_qwen38"), ("claude_sonnet_5_5 zero-shot", "pf_test_claude_sonnet_5_5")]:
        best = {}
        for line in (EXP / "vlm_conditions" / run / "raw.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["evidence"] == "postfire_image" and (r["global_id"] not in best or r.get("parse_ok")):
                best[r["global_id"]] = r
        scores[name] = {g: r["p_destroyed"] for g, r in best.items() if r.get("parse_ok")}
    for name, m in [("SFT Qwen3.5-4B", "Qwen3.5-4B"), ("SFT Qwen3.5-9B", "Qwen3.5-9B")]:
        log = json.loads((PF / f"sft_{m}" / "log.json").read_text())
        scores[name] = dict(zip(log["test_ids"], np.load(PF / f"sft_{m}" / "test_probs.npy")[:, 3:].sum(1)))
    lp = json.loads((PF / "lp_test_preds.json").read_text())
    scores["frozen DINOv2 LP (post)"] = dict(zip(lp["global_id"], lp["lp_post facebook/dinov2-base"]))

    gi = {g: i for i, g in enumerate(gid)}
    rng = np.random.default_rng(SEED)

    def auc(ids, s):
        idx = [gi[g] for g in ids]
        return roc_auc_score(yt[idx], [s[g] for g in ids])

    def boot(ids, s1, s2=None, n=1000):
        evs = np.array([ev[gi[g]] for g in ids])
        u = np.unique(evs)
        byev = {e: [g for g, x in zip(ids, evs) if x == e] for e in u}
        vals = []
        for _ in range(n):
            pick = [g for e in rng.choice(u, len(u)) for g in byev[e]]
            if len({yt[gi[g]] for g in pick}) < 2:
                continue
            vals.append(auc(pick, s1) - (auc(pick, s2) if s2 else 0))
        return np.percentile(vals, [2.5, 97.5]).tolist()

    L = ["# Retrieval-augmented baselines (post-fire, 2024+ test events)\n"]
    res = {"rows": [], "tests": []}
    for sname, ids_all in [("subset400", sorted(sub & set(gid))), ("full", gid)]:
        L += [f"\n## {sname} ({len(ids_all)} buildings)\n", "| Model | N | AUROC [95% event CI] |", "|---|---:|---|"]
        for name, s in scores.items():
            ids = [g for g in ids_all if g in s]
            if len(ids) < 0.95 * len(ids_all):
                continue
            a, ci = auc(ids, s), boot(ids, s)
            res["rows"].append({"set": sname, "model": name, "n": len(ids), "auroc": a, "ci95": ci})
            L.append(f"| {name} | {len(ids)} | {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    L += ["\n## Pre-registered tests (subset400, same buildings)\n", "| Test | A − B | Δ AUROC | 95% CI |", "|---|---|---:|---|"]
    qn = "qwen3_8_27b"
    for tag, a, b in [("R1", f"{qn} few-shot retrieved", "qwen3_8_27b zero-shot"),
                      ("R2", f"{qn} few-shot retrieved", f"{qn} few-shot random"),
                      ("R3", f"{qn} few-shot retrieved", "kNN-4 unweighted vote"),
                      ("R4", f"{qn} few-shot retrieved", "SFT Qwen3.5-4B"),
                      ("R4", f"{qn} few-shot retrieved", "frozen DINOv2 LP (post)"),
                      ("R4", "claude_sonnet_5_5 few-shot retrieved", "claude_sonnet_5_5 zero-shot"),
                      ("R4", "claude_sonnet_5_5 few-shot retrieved", "kNN-4 unweighted vote"),
                      ("R4", "claude_sonnet_5_5 few-shot retrieved", "SFT Qwen3.5-9B")]:
        if a not in scores or b not in scores:
            continue
        ids = [g for g in sorted(sub & set(gid)) if g in scores[a] and g in scores[b]]
        d, ci = auc(ids, scores[a]) - auc(ids, scores[b]), boot(ids, scores[a], scores[b])
        res["tests"].append({"test": tag, "a": a, "b": b, "n": len(ids), "delta": d, "ci95": ci})
        L.append(f"| {tag} | {a} − {b} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    (OUT / "report.json").write_text(json.dumps(res, indent=2))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--conditions", nargs="+", default=["retrieved", "random"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--knn-only", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    mem, Em, ym, te, sims = setup()
    if args.model:
        run_vlm(args, mem, ym, te, sims)
    if args.report or args.knn_only:
        report(mem, ym, te, sims)


if __name__ == "__main__":
    main()
