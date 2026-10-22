"""
How much does the evaluation split inflate scores?  (exploratory methodology analysis)

Same models, same pooled data, three protocols:
  event/time split   train on 2019-2023 train(+val) events, test on 2024+ events (our protocol)
  random split       buildings shuffled at random into the same train/test sizes (10 seeds)
  random within test events  train and test drawn from the SAME 2024+ fires (10 seeds):
                     the "same fire seen in training" leak typical of patch-level random splits

Tasks / models
  post-fire  frozen DINOv2-base (cached) + PCA-128 + LR on post-fire crops      (postfire_*.jsonl)
  pre-fire   HGB on DINS pre-fire metadata, and on inspector attributes (leaky)  (all residential 2019+)

Output: output/experiments/split_inflation.{json,md}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.tabular_baselines_on_pilot import INSPECTOR, SPLITS, SRC, model, prep  # noqa: E402
from scripts.train.sft_postfire_qwen import load  # noqa: E402

PF = ROOT / "output" / "experiments" / "postfire"
OUT = ROOT / "output" / "experiments"
SEEDS = range(10)


def lp():
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=128, random_state=0)),
                     ("lr", LogisticRegression(C=0.01, max_iter=5000))])


def postfire_rows():
    recs, E = [], []
    for s in ("train", "val", "test"):
        r = load(s)
        z = np.load(PF / f"emb_dinov2base_{s}.npz", allow_pickle=True)
        assert list(z["ids"]) == [x["global_id"] for x in r]
        d = z["E"].shape[1] // 2
        recs += r
        E.append(z["E"][:, :d])
    E = np.concatenate(E)
    y = np.array([x["labels"]["outcome"] == "Destroyed" for x in recs], dtype=int)
    split = np.array([x["split"] for x in recs])
    return E, y, split


def protocols(n_tr_mask, te_mask, fit_eval, y, rng_pool_mask=None):
    """Return dict protocol -> list of AUROCs."""
    out = {"event/time split": [fit_eval(n_tr_mask, te_mask)]}
    n_tr, n_te = int(n_tr_mask.sum()), int(te_mask.sum())
    pool = np.where(n_tr_mask | te_mask)[0]
    out["random split"] = []
    for s in SEEDS:
        p = np.random.default_rng(s).permutation(pool)
        tr, te = np.zeros(len(y), bool), np.zeros(len(y), bool)
        tr[p[:n_tr]], te[p[n_tr:n_tr + n_te]] = True, True
        out["random split"].append(fit_eval(tr, te))
    tpool = np.where(te_mask)[0]
    out["random within test events (same fires in train)"] = []
    for s in SEEDS:
        p = np.random.default_rng(100 + s).permutation(tpool)
        k = len(p) // 2
        tr, te = np.zeros(len(y), bool), np.zeros(len(y), bool)
        tr[p[:k]], te[p[k:]] = True, True
        out["random within test events (same fires in train)"].append(fit_eval(tr, te))
    return out


def main() -> None:
    res = {}
    E, y, split = postfire_rows()

    def fe_post(tr, te):
        m = lp().fit(E[tr], y[tr])
        return float(roc_auc_score(y[te], m.predict_proba(E[te])[:, 1]))
    res["post-fire | DINOv2 LP"] = protocols(np.isin(split, ["train", "val"]), split == "test", fe_post, y)
    print("post-fire done", flush=True)

    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    yy = df["y"].values
    for name, cat in [("pre-fire | metadata HGB", ["STRUCTURETYPE"]),
                      ("pre-fire | inspector attrs HGB (leaky)", ["STRUCTURETYPE"] + INSPECTOR)]:
        cols = cat + ["YEARBUILT", "log_value"]

        def fe(tr, te, cat=cat, cols=cols):
            m = model("hgb", cat, ["YEARBUILT", "log_value"]).fit(df.loc[tr, cols], yy[tr])
            return float(roc_auc_score(yy[te], m.predict_proba(df.loc[te, cols])[:, 1]))
        res[name] = protocols(df["split"].isin(["train", "val"]).values, (df["split"] == "test").values, fe, yy)
        print(name, "done", flush=True)

    L = ["# Split inflation (exploratory): same model, same data, different evaluation protocol\n",
         "AUROC; random protocols are mean ± sd over 10 seeds (min–max).\n",
         "| Task / model | Event/time split (ours) | Random building split | Random within test fires | Inflation (random − event) |",
         "|---|---:|---:|---:|---:|"]
    for k, v in res.items():
        e = v["event/time split"][0]
        r = np.array(v["random split"])
        w = np.array(v["random within test events (same fires in train)"])
        L.append(f"| {k} | {e:.3f} | {r.mean():.3f} ± {r.std():.3f} ({r.min():.3f}–{r.max():.3f}) | "
                 f"{w.mean():.3f} ± {w.std():.3f} | {r.mean() - e:+.3f} |")
    (OUT / "split_inflation.json").write_text(json.dumps(res, indent=2))
    (OUT / "split_inflation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
