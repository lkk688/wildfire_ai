"""
(A2) Resolution ablation at a fixed ~75 m field of view: zoom-18 centre 256 px (~0.5 m/px)
vs zoom-19 full 512 px (~0.25 m/px, native capture), same buildings and fit/eval protocol
(frozen DINOv2-base + PCA-64 + LR fitted on the pre-fire train sample; T1 AUROC, T2 pair acc).
Output: output/experiments/resolution_ablation.json
Run in the GPU env.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.context_ablation import jl, lr  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
Z19 = json.loads((SPL / "z19_manifest.json").read_text())


@torch.no_grad()
def emb(paths, crop, proc, model, bs=48):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        if crop:
            ims = [im.crop((128, 128, 384, 384)) for im in ims]
        x = proc(images=ims, return_tensors="pt").to("cuda")
        out.append(model(**x).pooler_output.float().cpu().numpy())
    return np.concatenate(out)


def main():
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    ok = lambda r: Z19.get(r["global_id"], {}).get("accepted")  # noqa: E731
    tr = [r for r in jl(SPL / "train_sample.jsonl") if r["imagery"]["accepted"] and ok(r)]
    t1 = [r for r in jl(SPL / "pilot_test.jsonl") if r["imagery"]["accepted"] and r["pilot_set"] == "diagnostic" and ok(r)]
    pairs = [q for q in jl(SPL / "pairs_test.jsonl") + jl(SPL / "pairs_test_large.jsonl") if ok(q["destroyed"]) and ok(q["survived"])]
    y = lambda rs: np.array([r["labels"]["outcome"] == "Destroyed" for r in rs], dtype=int)  # noqa: E731
    res = {"n_train": len(tr), "n_t1": len(t1), "n_pairs": len(pairs)}
    for name, path_fn, crop in [("z18_center_75m", lambda r: r["imagery"]["image_path"], True),
                                ("z19_native_75m", lambda r: Z19[r["global_id"]]["image_path"], False)]:
        m = lr().fit(emb([path_fn(r) for r in tr], crop, proc, model), y(tr))
        p1 = m.predict_proba(emb([path_fn(r) for r in t1], crop, proc, model))[:, 1]
        pd_ = m.predict_proba(emb([path_fn(q["destroyed"]) for q in pairs], crop, proc, model))[:, 1]
        ps_ = m.predict_proba(emb([path_fn(q["survived"]) for q in pairs], crop, proc, model))[:, 1]
        acc = (pd_ > ps_).astype(float)
        y1, ev1, evp = y(t1), np.array([r["event"] for r in t1]), np.array([q["event"] for q in pairs])
        res[name] = {"T1_auroc": float(roc_auc_score(y1, p1)),
                     "T1_ci": _cluster_boot(ev1, lambda i: roc_auc_score(y1[i], p1[i]) if len(set(y1[i])) == 2 else np.nan),
                     "T2_pair_acc": float(acc.mean()), "T2_ci": _cluster_boot(evp, lambda i: acc[i].mean())}
        print(name, res[name], flush=True)
    (ROOT / "output" / "experiments" / "resolution_ablation.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
