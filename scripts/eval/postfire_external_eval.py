"""
Out-of-California test of post-fire damage recognition: the 2021 Marshall Fire (Colorado), BRIGHT building labels,
capture-date-verified Wayback imagery (scripts/data/build_marshall_external.py). Models are trained only on DINS
(California) data and are not tuned on this set.

  - frozen DINOv2-base linear probe (post, and pre+post), same configuration as the main post-fire table,
    refit on the DINS post-fire train+val embeddings
  - Qwen3.5-4B / 9B + LoRA adapters from scripts/train/sft_postfire_qwen.py (token readout p(major)+p(destroyed))
CI: bootstrap over BRIGHT tiles (spatial clusters; a single fire).
Output: output/experiments/postfire/external_marshall.json. Run in the GPU env.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

PF = ROOT / "output" / "experiments" / "postfire"
EXT = ROOT / "output" / "splits" / "postfire_ext_marshall.jsonl"


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def summarise(y, p, clusters):
    return {"auroc": float(roc_auc_score(y, p)), "n": int(len(y)), "pos_rate": float(np.mean(y)),
            "ci95": _cluster_boot(clusters, lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=str(EXT))
    ap.add_argument("--adapters", nargs="*", default=["sft_Qwen3.5-4B:Qwen/Qwen3.5-4B", "sft_Qwen3.5-9B:Qwen/Qwen3.5-9B"])
    a = ap.parse_args()
    ext = jl(a.ext)
    y = np.array([r["labels"]["outcome"] == "Destroyed" for r in ext], dtype=int)
    cl = np.array([r.get("bright_tile", r["event"]) for r in ext])
    res = {"set": Path(a.ext).name, "n": len(ext)}

    import torch
    from scripts.eval.embedding_baseline import embed
    from scripts.train.sft_postfire_qwen import load
    tr, va = load("train"), load("val")
    E = {}
    for tag, recs in (("train", tr), ("val", va)):
        z = np.load(PF / f"emb_dinov2base_{tag}.npz", allow_pickle=True)
        assert list(z["ids"]) == [r["global_id"] for r in recs], f"embedding cache out of date for {tag}"
        E[tag] = z["E"]
    post = embed([r["imagery_postfire"]["image_path"] for r in ext], "facebook/dinov2-base", "cuda")
    pre = embed([r["imagery"]["image_path"] for r in ext], "facebook/dinov2-base", "cuda")
    Ex = np.concatenate([post, pre], 1)
    Etr = np.concatenate([E["train"], E["val"]])
    ytr = np.array([r["labels"]["outcome"] == "Destroyed" for r in tr + va], dtype=int)
    d = Etr.shape[1] // 2
    for name, f in (("lp_post", lambda M: M[:, :d]), ("lp_prepost", lambda M: np.concatenate([M, M[:, :d] - M[:, d:]], 1))):
        clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=128, random_state=0)),
                        ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=4000, scoring="roc_auc"))])
        clf.fit(f(Etr), ytr)
        res[name] = summarise(y, clf.predict_proba(f(Ex))[:, 1], cl)
        print(name, res[name], flush=True)
    torch.cuda.empty_cache()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from scripts.train.sft_postfire_qwen import Batcher, score
    for spec in a.adapters:
        run, model_id = spec.split(":")
        ad = PF / run / "adapter"
        if not ad.exists():
            continue
        proc = AutoProcessor.from_pretrained(model_id)
        proc.tokenizer.padding_side = "right"
        base = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
        b = Batcher(proc)
        pb = score(base, b, ext)[:, 3:].sum(1)
        res[f"{run}_base_readout"] = summarise(y, pb, cl)
        model = PeftModel.from_pretrained(base, ad).eval()
        ps = score(model, b, ext)[:, 3:].sum(1)
        res[run] = summarise(y, ps, cl)
        print(run, res[run], "base", res[f"{run}_base_readout"]["auroc"], flush=True)
        del model, base
        torch.cuda.empty_cache()
    (PF / f"external_{Path(a.ext).stem.replace('postfire_ext_', '')}.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
