"""
(C1) Larger pre-fire evaluation with free/local models.

Sets (2024+ test fires, disjoint from the pilot and earlier pair sets):
  T2-XL  output/splits/pairs_test_xl.jsonl     same-fire neighbour pairs (15-100 m)
  T1-XL  output/splits/test_natural_xl.jsonl   buildings at natural prevalence
Models (all fitted on 2019-23 fires only):
  tabular HGB (type, year built, value) | frozen DINOv2 two-view LR probe (971-building sample)
  | DINOv2 pair ranker (747 train pairs) | Qwen3.5-4B pre-fire LoRA (token readout)
Output: output/experiments/c1_scaleup.json
Run in the GPU env.
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
from scripts.eval.pair_trained_baseline import embed_all as embed3  # noqa: E402
from scripts.eval.tabular_baselines_on_pilot import SPLITS, SRC, model, prep  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
EXP = ROOT / "output" / "experiments"


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def pair_metrics(sd, ss, ev):
    acc = (sd > ss) + 0.5 * (sd == ss)
    return {"pair_acc": float(acc.mean()), "ci95": _cluster_boot(ev, lambda i: acc[i].mean()), "n": int(len(acc))}


def auc_metrics(y, p, ev):
    return {"auroc": float(roc_auc_score(y, p)), "n": int(len(y)), "pos_rate": float(np.mean(y)),
            "ci95": _cluster_boot(ev, lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan)}


def main():
    pairs = jl(SPL / "pairs_test_xl.jsonl")
    t1 = [r for r in jl(SPL / "test_natural_xl.jsonl") if r["imagery"]["accepted"]]
    pev = np.array([q["event"] for q in pairs])
    t1ev = np.array([r["event"] for r in t1])
    y1 = np.array([r["labels"]["outcome"] == "Destroyed" for r in t1], dtype=int)
    gd = [q["destroyed"]["global_id"] for q in pairs]
    gs = [q["survived"]["global_id"] for q in pairs]
    res = {"n_pairs": len(pairs), "n_t1": len(t1), "pair_events": sorted(set(pev.tolist()))}

    # tabular
    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    tr = df[df["split"].isin(["train", "val"])]
    D = df.set_index("GLOBALID")
    cols = ["STRUCTURETYPE", "YEARBUILT", "log_value"]
    m = model("hgb", ["STRUCTURETYPE"], ["YEARBUILT", "log_value"]).fit(tr[cols], tr["y"])
    res["tabular"] = {"T2": pair_metrics(m.predict_proba(D.loc[gd][cols])[:, 1], m.predict_proba(D.loc[gs][cols])[:, 1], pev),
                      "T1": auc_metrics(y1, m.predict_proba(D.loc[[r["global_id"] for r in t1]][cols])[:, 1], t1ev)}
    print("tabular", res["tabular"], flush=True)

    # DINOv2 two-view probe (same config as the main table)
    trs = [r for r in jl(SPL / "train_sample.jsonl") if r["imagery"]["accepted"]]
    path = {r["global_id"]: r["imagery"]["image_path"] for r in trs + t1}
    for q in pairs:
        for s in ("destroyed", "survived"):
            path[q[s]["global_id"]] = q[s]["imagery"]["image_path"]
    ids = list(path)
    E = embed([path[g] for g in ids], "facebook/dinov2-base", "cuda")
    ix = {g: i for i, g in enumerate(ids)}
    clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0)),
                    ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=3000, scoring="roc_auc"))])
    clf.fit(E[[ix[r["global_id"]] for r in trs]], [int(r["labels"]["outcome"] == "Destroyed") for r in trs])
    p = clf.predict_proba(E)[:, 1]
    res["dinov2_probe"] = {"T2": pair_metrics(p[[ix[g] for g in gd]], p[[ix[g] for g in gs]], pev),
                           "T1": auc_metrics(y1, p[[ix[r["global_id"]] for r in t1]], t1ev)}
    print("probe", res["dinov2_probe"], flush=True)

    # pair ranker on 3-scale DINOv2 features, trained on train pairs
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dm = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    trp = jl(SPL / "train_pairs.jsonl")
    tids = sorted({q[s]["global_id"] for q in trp for s in ("destroyed", "survived")})
    tpath = {q[s]["global_id"]: q[s]["imagery"]["image_path"] for q in trp for s in ("destroyed", "survived")}
    Et = embed3([tpath[g] for g in tids], proc, dm)
    Ex = embed3([path[g] for g in gd + gs], proc, dm)
    pca = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0))]).fit(Et)
    Zt, Zx = pca.transform(Et), pca.transform(Ex)
    tix = {g: i for i, g in enumerate(tids)}
    rng = np.random.default_rng(1019)
    X, yy = [], []
    for q in trp:
        a, b = tix[q["destroyed"]["global_id"]], tix[q["survived"]["global_id"]]
        if rng.random() < 0.5:
            X.append(Zt[b] - Zt[a]); yy.append(0)
        else:
            X.append(Zt[a] - Zt[b]); yy.append(1)
    rk = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=5000, fit_intercept=False).fit(np.array(X), yy)
    n = len(gd)
    sdiff = rk.decision_function(Zx[:n] - Zx[n:])
    res["dinov2_ranker"] = {"T2": pair_metrics(sdiff, np.zeros(n), pev)}
    print("ranker", res["dinov2_ranker"], flush=True)
    del dm
    torch.cuda.empty_cache()

    # pre-fire LoRA (token readout)
    adapter = EXP / "prefire_sft_Qwen3.5-4B" / "adapter"
    if adapter.exists():
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from scripts.train.sft_prefire_qwen import Batcher, score
        qp = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
        qp.tokenizer.padding_side = "right"
        qm = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16).cuda(),
                                       adapter).eval()
        b = Batcher(qp)
        sd = score(qm, b, [q["destroyed"] for q in pairs])
        ss = score(qm, b, [q["survived"] for q in pairs])
        s1 = score(qm, b, t1)
        res["prefire_lora_4b"] = {"T2": pair_metrics(sd, ss, pev), "T1": auc_metrics(y1, s1, t1ev)}
        print("lora", res["prefire_lora_4b"], flush=True)
    (EXP / "c1_scaleup.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
