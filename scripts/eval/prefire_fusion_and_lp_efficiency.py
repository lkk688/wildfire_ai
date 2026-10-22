"""
(C2) Best pre-fire model from ALL pre-fire information on exposure-controlled pairs:
     pair-level logistic fusion of score differences from
       tab   tabular pre-fire metadata (HGB, fitted on 2019-23 train(+val) events)
       emb   frozen DINOv2 LP score (fitted on the 971-building train sample)
       vlm   VLM pair judgment logit (Sonnet / Qwen3.8-27B, image condition)
     fused with event-grouped CV over pairs (no test labels used to fit components).
(B1-LP) Data efficiency of the post-fire frozen-DINOv2 linear probe with the same
     seeded stratified train subsets as the SFT runs (n = 100/400/800/all, 5 seeds).
Outputs: output/experiments/prefire_fusion.md, output/experiments/postfire/lp_efficiency.json
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.tabular_baselines_on_pilot import SPLITS, SRC, model, prep  # noqa: E402
from scripts.train.sft_postfire_qwen import load  # noqa: E402

EXP = ROOT / "output" / "experiments"
SPL = ROOT / "output" / "splits"


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def fusion():
    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    D = df.set_index("GLOBALID")
    cols = ["STRUCTURETYPE", "YEARBUILT", "log_value"]
    z = np.load(EXP / "embedding_baseline" / "emb_facebook__dinov2-base.npz", allow_pickle=True)
    E, eids = z["E"], list(z["ids"])
    # add embeddings for confirmatory-pair buildings that are not in the cache
    need = {}
    for l in (SPL / "confirm_pairs.jsonl").read_text().splitlines():
        q = json.loads(l)
        for side in ("destroyed", "survived"):
            if q[side]["global_id"] not in set(eids):
                need[q[side]["global_id"]] = q[side]["imagery"]["image_path"]
    if need:
        import torch
        from scripts.eval.embedding_baseline import embed
        extra = embed(list(need.values()), "facebook/dinov2-base", "cuda" if torch.cuda.is_available() else "cpu")
        E = np.concatenate([E, extra])
        eids = eids + list(need)
    eidx = {g: i for i, g in enumerate(eids)}
    tr_s = [json.loads(l) for l in (SPL / "train_sample.jsonl").read_text().splitlines()]
    tr_s = [r for r in tr_s if r["imagery"]["accepted"] and r["global_id"] in eidx]
    emb_m = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0)),
                      ("lr", LogisticRegression(C=0.01, max_iter=4000))])
    emb_m.fit(E[[eidx[r["global_id"]] for r in tr_s]], [int(r["labels"]["outcome"] == "Destroyed") for r in tr_s])
    L = ["# Pre-fire information fusion on exposure-controlled pairs\n",
         "Pair accuracy; components fitted on earlier fires only; fusion weights by event-grouped 5-fold CV over pairs.\n",
         "| Pair set | VLM | tab | emb | vlm | tab+emb | tab+vlm | tab+emb+vlm |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for pset, files, train_splits in [("test 2024-25 (403)", ["pairs_test.jsonl", "pairs_test_large.jsonl"], ["train", "val"]),
                                      ("val replication (289)", ["confirm_pairs.jsonl"], ["train"])]:
        tr = df[df["split"].isin(train_splits)]
        tab_m = model("hgb", ["STRUCTURETYPE"], ["YEARBUILT", "log_value"]).fit(tr[cols], tr["y"])
        pairs = {json.loads(l)["pair_id"]: json.loads(l) for f in files for l in (SPL / f).read_text().splitlines()}
        for vlm_name, runs in [("Sonnet", {"test": "pairs_sonnet", "val": "confirm_pairs_sonnet"}),
                               ("Qwen3.8-27B", {"test": "pairs_qwen38", "val": "confirm_pairs_qwen38"})]:
            run = runs["test" if pset.startswith("test") else "val"]
            best = {}
            for line in (EXP / "vlm_pairs" / run / "raw.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["evidence"] == "image" and r.get("parse_ok") and r["pair_id"] in pairs:
                    best[r["pair_id"]] = r
            rows = []
            for pid, r in best.items():
                q = pairs[pid]
                a, b = (q["destroyed"], q["survived"]) if r["destroyed_is"] == "A" else (q["survived"], q["destroyed"])
                if a["global_id"] not in eidx or b["global_id"] not in eidx:
                    continue
                ta, tb = tab_m.predict_proba(D.loc[[a["global_id"], b["global_id"]]][cols])[:, 1]
                ea, eb = emb_m.predict_proba(E[[eidx[a["global_id"]], eidx[b["global_id"]]]])[:, 1]
                rows.append({"y": int(r["destroyed_is"] == "A"), "tab": logit(ta) - logit(tb), "emb": logit(ea) - logit(eb),
                             "vlm": logit(r["p_A_destroyed"]), "event": q["event"]})
            X = pd.DataFrame(rows)
            accs = {}
            for feats in (["tab"], ["emb"], ["vlm"], ["tab", "emb"], ["tab", "vlm"], ["tab", "emb", "vlm"]):
                p = np.zeros(len(X))
                for tri, tei in GroupKFold(n_splits=min(5, X["event"].nunique())).split(X, X["y"], X["event"]):
                    m = LogisticRegression(fit_intercept=False).fit(X.iloc[tri][feats], X["y"].iloc[tri])
                    p[tei] = m.predict_proba(X.iloc[tei][feats])[:, 1]
                accs["+".join(feats)] = float(np.mean((p > 0.5) == X["y"]))
            L.append(f"| {pset} | {vlm_name} | " + " | ".join(f"{accs[k]:.3f}" for k in
                     ["tab", "emb", "vlm", "tab+emb", "tab+vlm", "tab+emb+vlm"]) + f" |  n={len(X)}")
    (EXP / "prefire_fusion.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def lp_efficiency():
    tr, va, te = load("train"), load("val"), load("test")
    emb = {}
    for s in ("train", "val", "test"):
        z = np.load(EXP / "postfire" / f"emb_dinov2base_{s}.npz", allow_pickle=True)
        d = z["E"].shape[1] // 2
        emb.update(dict(zip(z["ids"], z["E"][:, :d])))
    yte = np.array([r["labels"]["outcome"] == "Destroyed" for r in te], dtype=int)
    Xte = np.stack([emb[r["global_id"]] for r in te])
    res = {}
    for n in (100, 400, 800, 0):
        vals = []
        for seed in range(5):
            sub = tr
            if n:
                rs = random.Random(1000 + seed)
                by = {}
                for r in tr:
                    by.setdefault(r["labels"]["outcome"], []).append(r)
                frac = n / len(tr)
                sub = [r for v in by.values() for r in rs.sample(v, max(1, round(frac * len(v))))]
            X = np.stack([emb[r["global_id"]] for r in sub])
            y = np.array([r["labels"]["outcome"] == "Destroyed" for r in sub], dtype=int)
            m = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(128, len(sub) - 1), random_state=0)),
                          ("lr", LogisticRegression(C=0.01, max_iter=5000))]).fit(X, y)
            vals.append(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])))
            if not n:
                break
        res[str(n or len(tr))] = vals
        print("LP n=", n or len(tr), np.round(vals, 3))
    (EXP / "postfire" / "lp_efficiency.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    fusion()
    lp_efficiency()
