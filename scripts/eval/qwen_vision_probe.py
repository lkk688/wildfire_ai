"""
(B2) Where does the SFT gain come from?  Linear probe on the frozen vision tower of
the same VLM (Qwen3.5-4B / 9B): mean-pooled image embeddings after the vision merger,
PCA-128 + logistic regression, trained on post-fire train buildings, scored on test.
If this matches SFT (~0.89), the language model adds little beyond its own vision features.
Output: output/experiments/postfire/qwen_vision_probe.json
Run in the GPU env.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.sft_postfire_qwen import load  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

OUT = ROOT / "output" / "experiments" / "postfire"


@torch.no_grad()
def feats(model, proc, recs, bs=16):
    out = []
    for i in range(0, len(recs), bs):
        ims = [Image.open(r["imagery_postfire"]["image_path"]).convert("RGB") for r in recs[i:i + bs]]
        enc = proc.image_processor(images=ims, return_tensors="pt")
        pv = enc["pixel_values"].to(model.device, dtype=model.dtype)
        thw = enc["image_grid_thw"].to(model.device)
        emb = model.model.get_image_features(pixel_values=pv, image_grid_thw=thw)
        if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
            emb = emb.pooler_output
        if isinstance(emb, (list, tuple)):
            pooled = [e.float().mean(0) for e in emb]
        else:
            sizes = (thw.prod(-1) // (proc.image_processor.merge_size ** 2)).tolist()
            pooled = [e.float().mean(0) for e in torch.split(emb, sizes)]
        out.append(torch.stack(pooled).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    res = {}
    tr, va, te = load("train"), load("val"), load("test")
    y = lambda rs: np.array([r["labels"]["outcome"] == "Destroyed" for r in rs], dtype=int)  # noqa: E731
    ev = np.array([r["event"] for r in te])
    for name in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"):
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
        Xtr, Xte = feats(model, proc, tr + va), feats(model, proc, te)
        del model
        torch.cuda.empty_cache()
        m = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=128, random_state=0)),
                      ("lr", LogisticRegression(C=0.01, max_iter=5000))]).fit(Xtr, y(tr + va))
        p = m.predict_proba(Xte)[:, 1]
        yt = y(te)
        res[name] = {"auroc": float(roc_auc_score(yt, p)), "dim": int(Xtr.shape[1]),
                     "ci95": _cluster_boot(ev, lambda i: roc_auc_score(yt[i], p[i]) if len(set(yt[i])) == 2 else np.nan)}
        print(name, res[name], flush=True)
    (OUT / "qwen_vision_probe.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
