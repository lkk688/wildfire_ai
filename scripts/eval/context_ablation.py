"""
Context-vs-building ablation (paper analysis A1).

Where does the image signal live?  For pre-fire risk we hypothesise the
neighbourhood context (where the fire went); for post-fire damage, the building
itself.  Same frozen DINOv2-base + PCA-64 + logistic regression, fitted per
image variant on training-event buildings, evaluated on:
  T1  pre-fire single-building AUROC (pilot diagnostic, 238)
  T2  pre-fire same-fire neighbour pairs (403 pairs)
  T3  post-fire AUROC (1,246 test buildings)
Variants of each 512x512 crop (~150 m at zoom 18; ~0.3 m/px):
  full          the whole crop
  center_64m    central 224 px (building + immediate yard)
  center_19m    central 64 px (roof only)
  masked_19m    whole crop with a grey disk (r = 32 px, ~9.5 m) over the centre building
  masked_37m    grey disk r = 64 px (~19 m): building and yard removed
  ring_only     only the annulus between r = 64 px and 256 px
Output: output/experiments/context_ablation.{json,md}
Run in the GPU env.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
OUT = ROOT / "output" / "experiments"
GREY = (128, 128, 128)


def variant(im: Image.Image, v: str) -> Image.Image:
    c = 256
    if v == "full":
        return im
    if v.startswith("center_"):
        h = {"center_64m": 112, "center_19m": 32}[v]
        return im.crop((c - h, c - h, c + h, c + h))
    out = im.copy()
    d = ImageDraw.Draw(out)
    if v.startswith("masked_"):
        r = {"masked_19m": 32, "masked_37m": 64}[v]
        d.ellipse((c - r, c - r, c + r, c + r), fill=GREY)
        return out
    if v == "ring_only":
        d.ellipse((c - 64, c - 64, c + 64, c + 64), fill=GREY)
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, 511, 511), fill=255)
        bg = Image.new("RGB", im.size, GREY)
        return Image.composite(out, bg, mask)
    raise ValueError(v)


@torch.no_grad()
def embed(paths, v, proc, model, bs=48):
    out = []
    for i in range(0, len(paths), bs):
        ims = [variant(Image.open(p).convert("RGB"), v) for p in paths[i:i + bs]]
        x = proc(images=ims, return_tensors="pt").to("cuda")
        h = model(**x)
        out.append(h.pooler_output.float().cpu().numpy())
    return np.concatenate(out)


def lr():
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=64, random_state=0)),
                     ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=3000, scoring="roc_auc"))])


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def main() -> None:
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    pre_tr = [r for r in jl(SPL / "train_sample.jsonl") if r["imagery"]["accepted"]]
    t1 = [r for r in jl(SPL / "pilot_test.jsonl") if r["imagery"]["accepted"] and r["pilot_set"] == "diagnostic"]
    pairs = jl(SPL / "pairs_test.jsonl") + jl(SPL / "pairs_test_large.jsonl")
    pf = {s: [r for r in jl(SPL / f"postfire_{s}.jsonl") if r["imagery_postfire"]["accepted"] and r["imagery"]["accepted"]]
          for s in ("train", "val", "test")}
    y = lambda rs: np.array([r["labels"]["outcome"] == "Destroyed" for r in rs], dtype=int)  # noqa: E731
    variants = ["full", "center_64m", "center_19m", "masked_19m", "masked_37m", "ring_only"]
    res = {}
    for v in variants:
        Etr = embed([r["imagery"]["image_path"] for r in pre_tr], v, proc, model)
        E1 = embed([r["imagery"]["image_path"] for r in t1], v, proc, model)
        Ed = embed([p["destroyed"]["imagery"]["image_path"] for p in pairs], v, proc, model)
        Es = embed([p["survived"]["imagery"]["image_path"] for p in pairs], v, proc, model)
        m = lr().fit(Etr, y(pre_tr))
        p1 = m.predict_proba(E1)[:, 1]
        y1, ev1 = y(t1), np.array([r["event"] for r in t1])
        acc = (m.predict_proba(Ed)[:, 1] > m.predict_proba(Es)[:, 1]).astype(float)
        evp = np.array([p["event"] for p in pairs])
        Ptr = embed([r["imagery_postfire"]["image_path"] for r in pf["train"] + pf["val"]], v, proc, model)
        Pte = embed([r["imagery_postfire"]["image_path"] for r in pf["test"]], v, proc, model)
        m3 = lr().fit(Ptr, y(pf["train"] + pf["val"]))
        p3 = m3.predict_proba(Pte)[:, 1]
        y3, ev3 = y(pf["test"]), np.array([r["event"] for r in pf["test"]])
        res[v] = {
            "T1_auroc": float(roc_auc_score(y1, p1)),
            "T1_ci": _cluster_boot(ev1, lambda i: roc_auc_score(y1[i], p1[i]) if len(set(y1[i])) == 2 else np.nan),
            "T2_pair_acc": float(acc.mean()), "T2_ci": _cluster_boot(evp, lambda i: acc[i].mean()),
            "T3_auroc": float(roc_auc_score(y3, p3)),
            "T3_ci": _cluster_boot(ev3, lambda i: roc_auc_score(y3[i], p3[i]) if len(set(y3[i])) == 2 else np.nan)}
        print(v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in res[v].items()}, flush=True)
    (OUT / "context_ablation.json").write_text(json.dumps(res, indent=2, default=float))
    L = ["# Context-vs-building ablation (frozen DINOv2-base + LR, per-variant fit)\n",
         "| Image variant | T1 pre-fire AUROC | T2 pair acc. | T3 post-fire AUROC |", "|---|---|---|---|"]
    for v, r in res.items():
        L.append(f"| {v} | {r['T1_auroc']:.3f} [{r['T1_ci'][0]:.3f}, {r['T1_ci'][1]:.3f}] | "
                 f"{r['T2_pair_acc']:.3f} [{r['T2_ci'][0]:.3f}, {r['T2_ci'][1]:.3f}] | "
                 f"{r['T3_auroc']:.3f} [{r['T3_ci'][0]:.3f}, {r['T3_ci'][1]:.3f}] |")
    (OUT / "context_ablation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
