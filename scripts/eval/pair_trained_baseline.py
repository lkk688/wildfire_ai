"""
(Review W2) Pair-trained image baseline: is chance-level pair accuracy a property of the
imagery or of probes trained for absolute, cross-fire risk?

Train on same-fire destroyed/survived neighbour pairs from 2019-23 TRAIN fires
(output/splits/train_pairs.jsonl), never on test/val fires.  Features per building:
frozen DINOv2-base pooled embeddings at three scales (full ~150 m, centre ~64 m, roof ~19 m).
Model: logistic regression on the feature DIFFERENCE (a - b) with random pair orientation
(Bradley-Terry style ranking), PCA fitted on training buildings; variants add the
metadata difference.  Evaluate pair accuracy on T2 test (403) and replication (289).
Output: output/experiments/pair_trained_baseline.json
Run in the GPU env.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
SCALES = {"full": None, "c64": 112, "c19": 32}


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


@torch.no_grad()
def embed_all(paths, proc, model, bs=48):
    out = {k: [] for k in SCALES}
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        for k, h in SCALES.items():
            v = ims if h is None else [im.crop((256 - h, 256 - h, 256 + h, 256 + h)) for im in ims]
            x = proc(images=v, return_tensors="pt").to("cuda")
            out[k].append(model(**x).pooler_output.float().cpu().numpy())
    return np.concatenate([np.concatenate(out[k]) for k in SCALES], axis=1)


def meta(rec):
    pf = rec["inputs"]["prefire"]
    yb = pf.get("year_built") or np.nan
    v = pf.get("assessed_improved_value")
    return np.array([yb, np.log1p(v) if v else np.nan], dtype=float)


def main():
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    tr = jl(SPL / "train_pairs.jsonl")
    sets = {"test": jl(SPL / "pairs_test.jsonl") + jl(SPL / "pairs_test_large.jsonl"), "val": jl(SPL / "confirm_pairs.jsonl")}
    recs = {}
    for ps in [tr] + list(sets.values()):
        for q in ps:
            for s in ("destroyed", "survived"):
                recs[q[s]["global_id"]] = q[s]
    ids = list(recs)
    E = embed_all([recs[g]["imagery"]["image_path"] for g in ids], proc, model)
    idx = {g: i for i, g in enumerate(ids)}
    M = np.stack([meta(recs[g]) for g in ids])
    med = np.nanmedian(M[[idx[q[s]["global_id"]] for q in tr for s in ("destroyed", "survived")]], axis=0)
    M = np.where(np.isnan(M), med, M)
    tr_ids = sorted({q[s]["global_id"] for q in tr for s in ("destroyed", "survived")})
    pca = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0))]).fit(E[[idx[g] for g in tr_ids]])
    Z = pca.transform(E)
    msc = StandardScaler().fit(M[[idx[g] for g in tr_ids]])
    Mz = msc.transform(M)
    rng = np.random.default_rng(1019)

    def diffs(pairs, feat, orient=None):
        X, y = [], []
        for k, q in enumerate(pairs):
            a, b = idx[q["destroyed"]["global_id"]], idx[q["survived"]["global_id"]]
            flip = rng.random() < 0.5 if orient is None else False
            if flip:
                a, b = b, a
            X.append(feat[a] - feat[b])
            y.append(0 if flip else 1)
        return np.array(X), np.array(y)

    res = {"n_train_pairs": len(tr), "n_train_events": len({q["event"] for q in tr})}
    for name, feat in [("image (DINOv2, 3 scales)", Z), ("metadata", Mz), ("image + metadata", np.concatenate([Z, Mz], 1))]:
        Xtr, ytr = diffs(tr, feat)
        clf = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=5000, fit_intercept=False).fit(Xtr, ytr)
        out = {}
        for sname, ps in sets.items():
            X, _ = diffs(ps, feat, orient="fixed")  # destroyed first: correct iff score > 0
            acc = (clf.decision_function(X) > 0).astype(float)
            ev = np.array([q["event"] for q in ps])
            out[sname] = {"pair_acc": float(acc.mean()), "ci95": _cluster_boot(ev, lambda i: acc[i].mean()), "n": len(ps)}
        res[name] = out
        print(name, out, flush=True)
    (ROOT / "output" / "experiments" / "pair_trained_baseline.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
