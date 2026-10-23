"""
Non-VLM image baseline (Review 1019 section 7: "image encoder + classification head").

Frozen vision encoder -> CLS/pooled embedding of the centred pre-fire crop
(plus a centre crop of the inner 128 px, which zooms on the building) ->
logistic regression fitted on the train-event sample, evaluated on the pilot
test set (AUROC) and on the pair set (pair accuracy), with the same metrics as
the VLM runs.  If this matches the VLM, the VLM's value is not accuracy.

Run in the GPU env:
  /home/lkk/miniconda3/envs/grpo/bin/python scripts/eval/embedding_baseline.py --encoder facebook/dinov2-base
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
# main T2 set (403 pairs); the larger XL set is evaluated separately in c1_scaleup.py
MAIN_PAIR_FILES = [SPL / "pairs_test.jsonl", SPL / "pairs_test_large.jsonl"]
OUT = ROOT / "output" / "experiments" / "embedding_baseline"


def records():
    recs = {}
    for f, split in [("train_sample.jsonl", None), ("pilot_test.jsonl", None)]:
        for line in (SPL / f).read_text().splitlines():
            r = json.loads(line)
            if r["imagery"]["accepted"]:
                recs[r["global_id"]] = r
    pairs = [json.loads(line) for f in MAIN_PAIR_FILES for line in f.read_text().splitlines()]
    for q in pairs:
        for side in ("destroyed", "survived"):
            q[side]["split"] = "pairs"
            recs.setdefault(q[side]["global_id"], q[side])
    return recs, pairs


@torch.no_grad()
def embed(paths, encoder, device, bs=32):
    proc = AutoImageProcessor.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        feats = []
        for view in (ims, [im.crop((192, 192, 320, 320)) for im in ims]):
            x = proc(images=view, return_tensors="pt").to(device)
            if hasattr(model, "get_image_features"):  # dual-tower (CLIP / SigLIP): image tower only
                v = model.get_image_features(pixel_values=x["pixel_values"])
                v = v.pooler_output if hasattr(v, "pooler_output") else v
            else:
                h = model(**x)
                v = h.pooler_output if getattr(h, "pooler_output", None) is not None else h.last_hidden_state[:, 0]
            feats.append(v.float().cpu().numpy())
        out.append(np.concatenate(feats, axis=1))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="facebook/dinov2-base")
    args = ap.parse_args()
    recs, pairs = records()
    ids = sorted(recs)
    slug = args.encoder.replace("/", "__")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"emb_{slug}.npz"
    z = np.load(cache, allow_pickle=True) if cache.exists() else None
    if z is not None and set(ids) <= set(z["ids"]):
        E, ids = z["E"], list(z["ids"])
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        E = embed([recs[g]["imagery"]["image_path"] for g in ids], args.encoder, dev)
        np.savez(cache, E=E, ids=np.array(ids))
    idx = {g: i for i, g in enumerate(ids)}
    y = np.array([int(recs[g]["labels"]["outcome"] == "Destroyed") for g in ids])
    split = np.array([recs[g]["split"] for g in ids])
    ps = np.array([recs[g].get("pilot_set") for g in ids])
    ev = np.array([recs[g]["event"] for g in ids])
    tr = split == "train"
    clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0)),
                    ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=3000, scoring="roc_auc"))])
    clf.fit(E[tr], y[tr])
    p = clf.predict_proba(E)[:, 1]
    L = [f"# Embedding baseline: `{args.encoder}` (frozen) + PCA-64 + logistic regression\n",
         f"Fitted on {tr.sum()} train-event buildings from {len(set(ev[tr]))} events.\n",
         "| Evaluated on | N | AUROC / pair acc [95% event CI] |", "|---|---:|---|"]
    res = []
    for s in ("diagnostic", "natural"):
        m = (split == "test") & (ps == s)
        yy, pp, ee = y[m], p[m], ev[m]
        ci = _cluster_boot(ee, lambda i: roc_auc_score(yy[i], pp[i]) if len(set(yy[i])) == 2 else np.nan)
        res.append({"eval": f"pilot {s}", "n": int(m.sum()), "score": float(roc_auc_score(yy, pp)), "ci95": ci})
    acc = np.array([float(p[idx[q["destroyed"]["global_id"]]] > p[idx[q["survived"]["global_id"]]]) for q in pairs])
    pev = np.array([q["event"] for q in pairs])
    res.append({"eval": "pairs (pair accuracy)", "n": len(acc), "score": float(acc.mean()),
                "ci95": _cluster_boot(pev, lambda i: acc[i].mean())})
    for r in res:
        L.append(f"| {r['eval']} | {r['n']} | {r['score']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    md = "\n".join(L) + "\n"
    (OUT / f"summary_{slug}.md").write_text(md)
    (OUT / f"summary_{slug}.json").write_text(json.dumps(res, indent=2))
    print(md)


if __name__ == "__main__":
    main()
