"""
Output-form value of the post-fire VLM (docs/POSTFIRE_TASK_PLAN.md section 8).

A. 5-level grading: QWK, macro-F1, per-level recall, level MAE, 3-group accuracy.
B. Selective prediction (binary): threshold t picked on val (max balanced acc),
   confidence |p - t|; AURC, auto-label coverage at >= 95% / 98% accuracy,
   ECE (10 bins) and Brier, raw and after Platt scaling fitted on val.
Linear heads (frozen DINOv2-base, PCA-128, LR) are fitted on TRAIN only so that,
like SFT, val is used only for selection / calibration.

Run in the GPU env (embeddings are cached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  /home/lkk/miniconda3/envs/grpo/bin/python scripts/eval/postfire_output_value.py [--sft Qwen3.5-4B Qwen3.5-9B]
Output: output/experiments/postfire/output_value.{json,md}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.sft_postfire_qwen import LEVEL_MAP, LEVELS, load  # noqa: E402

EXP = ROOT / "output" / "experiments"
PF = EXP / "postfire"
SEED = 1019


# ---------------------------------------------------------------- predictions
def embeddings(recs, tag):
    cache = PF / f"emb_dinov2base_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == [r["global_id"] for r in recs]:
            return z["E"]
    import torch
    from scripts.eval.embedding_baseline import embed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    post = embed([r["imagery_postfire"]["image_path"] for r in recs], "facebook/dinov2-base", dev)
    pre = embed([r["imagery"]["image_path"] for r in recs], "facebook/dinov2-base", dev)
    E = np.concatenate([post, pre], axis=1)
    np.savez(cache, E=E, ids=np.array([r["global_id"] for r in recs]))
    return E


def linear_heads(tr, va, te):
    Etr, Eva, Ete = embeddings(tr, "train"), embeddings(va, "val"), embeddings(te, "test")
    d = Etr.shape[1] // 2
    lv = lambda rs: np.array([LEVELS.index(LEVEL_MAP[r["labels"]["damage"]]) for r in rs])  # noqa: E731
    out = {}
    for name, f in [("post", lambda E: E[:, :d]),
                    ("prepost", lambda E: np.concatenate([E[:, d:], E[:, :d], E[:, :d] - E[:, d:]], axis=1))]:
        clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=128, random_state=0)),
                        ("lr", LogisticRegression(C=0.01, max_iter=5000))])
        clf.fit(f(Etr), lv(tr))
        cls = list(clf.classes_)

        def full(P):
            Q = np.zeros((len(P), 5))
            for j, c in enumerate(cls):
                Q[:, c] = P[:, j]
            return Q
        out[f"LP5 dinov2 {name}"] = {"val": full(clf.predict_proba(f(Eva))), "test": full(clf.predict_proba(f(Ete)))}
    return out


def zero_shot(run, evidence, recs):
    p = EXP / "vlm_conditions" / run / "raw.jsonl"
    if not p.exists():
        return None
    best = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] == evidence and (r["global_id"] not in best or r.get("parse_ok")):
            best[r["global_id"]] = r
    return np.array([best[r["global_id"]]["p_destroyed"] if best.get(r["global_id"], {}).get("parse_ok") else np.nan
                     for r in recs])


# ---------------------------------------------------------------- metrics
def level_metrics(y5, P):
    pred = P.argmax(1)
    grp = lambda a: np.where(a == 0, 0, np.where(a <= 2, 1, 2))  # noqa: E731
    return {"qwk": float(cohen_kappa_score(y5, pred, weights="quadratic")),
            "macro_f1": float(f1_score(y5, pred, average="macro", labels=range(5))),
            "level_mae": float(np.mean(np.abs(pred - y5))),
            "acc_3group": float(np.mean(grp(pred) == grp(y5))),
            "recall_per_level": [float(x) for x in recall_score(y5, pred, average=None, labels=range(5), zero_division=0)]}


def pick_threshold(y, p):
    ts = np.unique(np.round(p, 4))
    best, bt = -1, 0.5
    for t in ts:
        yh = p >= t
        ba = 0.5 * (np.mean(yh[y == 1]) + np.mean(~yh[y == 0]))
        if ba > best:
            best, bt = ba, t
    return float(bt)


def selective(y, p, t):
    conf = np.abs(p - t)
    order = np.argsort(-conf, kind="stable")
    correct = ((p >= t).astype(int) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
    risk = 1 - cum
    aurc = float(np.mean(risk))
    cov = {}
    for a in (0.95, 0.98):
        ok = np.where(cum >= a)[0]
        cov[f"coverage_at_{int(a * 100)}"] = float((ok.max() + 1) / len(y)) if len(ok) else 0.0
    return {"aurc": aurc, **cov, "acc_full": float(cum[-1])}


def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins) if lo < 1 - 1 / bins else (p >= lo)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def platt(yv, pv, pt):
    lr = LogisticRegression().fit(np.log(np.clip(pv, 1e-4, 1 - 1e-4) / np.clip(1 - pv, 1e-4, 1)).reshape(-1, 1), yv)
    z = np.log(np.clip(pt, 1e-4, 1 - 1e-4) / np.clip(1 - pt, 1e-4, 1)).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", nargs="+", default=["Qwen3.5-4B"])
    args = ap.parse_args()
    tr, va, te = load("train"), load("val"), load("test")
    yv = np.array([r["labels"]["outcome"] == "Destroyed" for r in va], dtype=int)
    yt = np.array([r["labels"]["outcome"] == "Destroyed" for r in te], dtype=int)
    y5 = np.array([LEVELS.index(LEVEL_MAP[r["labels"]["damage"]]) for r in te])
    ev = np.array([r["event"] for r in te])

    five, binp = {}, {}
    for name, d in linear_heads(tr, va, te).items():
        five[name] = d["test"]
        binp[name] = (d["val"][:, 3:].sum(1), d["test"][:, 3:].sum(1))
    for m in args.sft:
        d = PF / f"sft_{m}"
        if not (d / "test_probs.npy").exists():
            print(f"[!] {m}: no test_probs yet, skipped")
            continue
        log = json.loads((d / "log.json").read_text())
        Pt, Pv = np.load(d / "test_probs.npy"), np.load(d / f"val_epoch{log['best_epoch']}.npy")
        five[f"SFT {m}"] = Pt
        binp[f"SFT {m}"] = (Pv[:, 3:].sum(1), Pt[:, 3:].sum(1))
        base_t = d / "test_base_probs.npy" if (d / "test_base_probs.npy").exists() else (
            PF / "sft_base_eval" / "test_base_probs.npy" if m == "Qwen3.5-4B" else None)
        if base_t is not None and base_t.exists():
            Bt, Bv = np.load(base_t), np.load(d / "val_epoch0.npy")
            five[f"base {m} (same readout)"] = Bt
            binp[f"base {m} (same readout)"] = (Bv[:, 3:].sum(1), Bt[:, 3:].sum(1))
    for name, run_v, run_t in [("Qwen3.8-27B zero-shot", "pf_val_qwen38", "pf_test_qwen38"),
                               ("Sonnet zero-shot", "pf_val_claude_sonnet_5_5", "pf_test_claude_sonnet_5_5"),
                               ("Haiku zero-shot", "pf_val_claude_haiku_5_5", "pf_test_claude_haiku_5_5")]:
        for evd in ("postfire_image", "prepost_image"):
            pv, pt = zero_shot(run_v, evd, va), zero_shot(run_t, evd, te)
            if pv is not None and pt is not None:
                binp[f"{name} {evd.split('_')[0]}"] = (pv, pt)

    sub = ~np.isnan(binp["Sonnet zero-shot postfire"][1]) if "Sonnet zero-shot postfire" in binp else None
    rng = np.random.default_rng(SEED)
    u = np.unique(ev)
    idx = {e: np.where(ev == e)[0] for e in u}
    boots = [np.concatenate([idx[e] for e in rng.choice(u, len(u))]) for _ in range(1000)]

    res = {"five_level": {}, "selective": {}, "paired": []}
    for name, P in five.items():
        res["five_level"][name] = level_metrics(y5, P)
    sel_fn = {}
    for name, (pv, pt) in binp.items():
        okv = ~np.isnan(pv)
        t = pick_threshold(yv[okv], pv[okv])
        pc = platt(yv[okv], pv[okv], np.nan_to_num(pt, nan=0.5))
        for sname, mask in [("full", ~np.isnan(pt)), ("subset400", sub if sub is not None else ~np.isnan(pt))]:
            if sname == "full" and "zero-shot" in name and "27B" not in name:
                continue
            m = mask & ~np.isnan(pt)
            if m.sum() < 50:
                continue
            r = selective(yt[m], pt[m], t)
            r.update({"n": int(m.sum()), "threshold_from_val": t, "auroc": float(roc_auc_score(yt[m], pt[m])),
                      "ece_raw": ece(yt[m], pt[m]), "brier_raw": float(np.mean((pt[m] - yt[m]) ** 2)),
                      "ece_platt": ece(yt[m], pc[m]), "brier_platt": float(np.mean((pc[m] - yt[m]) ** 2))})
            res["selective"][f"{sname} | {name}"] = r
        sel_fn[name] = (pt, t)

    def paired(a, b, fn, mask):
        vals = []
        for bi in boots:
            bi = bi[mask[bi]]
            vals.append(fn(a, bi) - fn(b, bi))
        return float(fn(a, np.where(mask)[0]) - fn(b, np.where(mask)[0])), np.percentile(vals, [2.5, 97.5]).tolist()

    ref5, refb = "LP5 dinov2 post", "LP5 dinov2 post"
    allm = np.ones(len(yt), bool)
    for m in [k for k in five if k.startswith("SFT")]:
        for met in ("qwk", "macro_f1"):
            f = lambda name, i, met=met: level_metrics(y5[i], five[name][i])[met]  # noqa: E731
            d, ci = paired(m, ref5, f, allm)
            res["paired"].append({"test": f"A {met}", "a": m, "b": ref5, "delta": d, "ci95": ci})
        for met in ("aurc", "coverage_at_95"):
            def f(name, i, met=met):
                pt, t = sel_fn[name]
                return selective(yt[i], pt[i], t)[met]
            d, ci = paired(m, refb, f, allm)
            res["paired"].append({"test": f"B {met}", "a": m, "b": refb, "delta": d, "ci95": ci})
        if sub is not None:
            for zs in ("Sonnet zero-shot postfire", "Sonnet zero-shot prepost", "Haiku zero-shot prepost"):
                if zs in sel_fn:
                    def f(name, i):
                        pt, t = sel_fn[name]
                        return selective(yt[i], pt[i], t)["coverage_at_95"]
                    d, ci = paired(m, zs, f, sub & ~np.isnan(sel_fn[zs][0]))
                    res["paired"].append({"test": "B coverage_at_95 (subset400)", "a": m, "b": zs, "delta": d, "ci95": ci})

    L = ["# Output-form value: 5-level grading and selective prediction (2024+ test events)\n",
         "## A. 5-level grading (full test, 1,246)\n",
         "| Model | QWK | Macro-F1 | Level MAE | 3-group acc | Recall: none / affected / minor / major / destroyed |",
         "|---|---:|---:|---:|---:|---|"]
    for k, v in res["five_level"].items():
        L.append(f"| {k} | {v['qwk']:.3f} | {v['macro_f1']:.3f} | {v['level_mae']:.3f} | {v['acc_3group']:.3f} | "
                 + " / ".join(f"{x:.2f}" for x in v["recall_per_level"]) + " |")
    L += ["\n## B. Selective prediction (threshold from val; confidence |p − t|)\n",
          "| Set | Model | N | AUROC | AURC ↓ | Auto-label coverage @95% acc | @98% | ECE raw / Platt | Brier raw / Platt |",
          "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for k, v in res["selective"].items():
        s, n = k.split(" | ")
        L.append(f"| {s} | {n} | {v['n']} | {v['auroc']:.3f} | {v['aurc']:.4f} | {v['coverage_at_95']:.2f} | "
                 f"{v['coverage_at_98']:.2f} | {v['ece_raw']:.3f} / {v['ece_platt']:.3f} | "
                 f"{v['brier_raw']:.3f} / {v['brier_platt']:.3f} |")
    L += ["\n## Pre-registered paired tests (SFT − reference, event bootstrap)\n",
          "| Test | A | B | Δ | 95% CI |", "|---|---|---|---:|---|"]
    for r in res["paired"]:
        L.append(f"| {r['test']} | {r['a']} | {r['b']} | {r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |")
    (PF / "output_value.json").write_text(json.dumps(res, indent=2))
    (PF / "output_value.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
