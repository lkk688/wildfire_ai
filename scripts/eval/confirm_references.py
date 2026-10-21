"""
H6 / H7 references for the confirmatory val-event sets (docs/CONFIRMATORY_PLAN.md).

H6  tabular HGB on pre-fire metadata, fitted on TRAIN events only (val excluded),
    pair accuracy on confirm_pairs (ties = half credit).
H7  frozen DINOv2-base + PCA-64 + LR fitted on the train-event image sample;
    AUROC on confirm_direct and pair accuracy on confirm_pairs.  The test-set
    finding to confirm: decent AUROC but ~chance pair accuracy.

Run in the GPU env:
  /home/lkk/miniconda3/envs/grpo/bin/python scripts/eval/confirm_references.py
Output: output/experiments/confirmatory/pair_references.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.embedding_baseline import embed  # noqa: E402
from scripts.eval.tabular_baselines_on_pilot import SPLITS, SRC, model, prep  # noqa: E402

SPL = ROOT / "output" / "splits"
OUT = ROOT / "output" / "experiments" / "confirmatory"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(line) for line in (SPL / "confirm_pairs.jsonl").read_text().splitlines()]
    direct = [json.loads(line) for line in (SPL / "confirm_direct.jsonl").read_text().splitlines()]
    res = {}

    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    tr = df[df["split"] == "train"]
    D = df.set_index("GLOBALID")
    cols = ["STRUCTURETYPE", "YEARBUILT", "log_value"]
    m = model("hgb", ["STRUCTURETYPE"], ["YEARBUILT", "log_value"]).fit(tr[cols], tr["y"])
    a = m.predict_proba(D.loc[[p["destroyed"]["global_id"] for p in pairs]][cols])[:, 1]
    b = m.predict_proba(D.loc[[p["survived"]["global_id"] for p in pairs]][cols])[:, 1]
    res["H6 tabular prefire_meta (train-only) pair acc"] = float(np.mean((a > b) + 0.5 * (a == b)))
    yd = np.array([r["labels"]["outcome"] == "Destroyed" for r in direct], dtype=int)
    pdir = m.predict_proba(D.loc[[r["global_id"] for r in direct]][cols])[:, 1]
    res["H6 tabular prefire_meta (train-only) AUROC confirm_direct"] = float(roc_auc_score(yd, pdir))

    train = [json.loads(line) for line in (SPL / "train_sample.jsonl").read_text().splitlines()]
    train = [r for r in train if r["imagery"]["accepted"]]
    ids = ([r["global_id"] for r in train] + [r["global_id"] for r in direct]
           + [p[s]["global_id"] for p in pairs for s in ("destroyed", "survived")])
    path = {r["global_id"]: r["imagery"]["image_path"] for r in train + direct}
    path.update({p[s]["global_id"]: p[s]["imagery"]["image_path"] for p in pairs for s in ("destroyed", "survived")})
    uniq = list(dict.fromkeys(ids))
    E = embed([path[g] for g in uniq], "facebook/dinov2-base", "cuda" if torch.cuda.is_available() else "cpu")
    idx = {g: i for i, g in enumerate(uniq)}
    Xtr = E[[idx[r["global_id"]] for r in train]]
    ytr = np.array([r["labels"]["outcome"] == "Destroyed" for r in train], dtype=int)
    clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0)),
                    ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=3000, scoring="roc_auc"))])
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(E)[:, 1]
    res["H7 DINOv2 head AUROC confirm_direct"] = float(roc_auc_score(yd, p[[idx[r['global_id']] for r in direct]]))
    pa = p[[idx[q["destroyed"]["global_id"]] for q in pairs]]
    pb = p[[idx[q["survived"]["global_id"]] for q in pairs]]
    res["H7 DINOv2 head pair acc confirm_pairs"] = float(np.mean(pa > pb))
    (OUT / "pair_references.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
