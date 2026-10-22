"""
Supervised image baselines for post-fire damage recognition (direction 2).

Any VLM result (zero-shot or SFT) must be read against these:
  lp_post      frozen encoder embedding of the post-fire crop (+ centre zoom) -> LR
  lp_prepost   [pre, post, post - pre] embeddings -> LR (change detection)
  ft_post      DINOv2-small fully fine-tuned on post-fire crops (flip/rot90 aug)
Fitted on postfire_train (2019-2023 train events), model selection on
postfire_val, reported on postfire_test (2024+ events): binary AUROC (Destroyed+
Major vs rest), balanced accuracy, and 5-level macro-F1 for the fine-tuned model.

Run in the GPU env:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  /home/lkk/miniconda3/envs/grpo/bin/python scripts/eval/postfire_supervised_baselines.py
Output: output/experiments/postfire/supervised_baselines.{json,md}
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.embedding_baseline import embed  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
OUT = ROOT / "output" / "experiments" / "postfire"
LEVELS = ["No Damage", "Affected (>0-10%)", "Minor (10-25%)", "Major (25-50%)", "Destroyed (>50%)"]


def load(split):
    recs = [json.loads(line) for line in (SPL / f"postfire_{split}.jsonl").read_text().splitlines()]
    return [r for r in recs if r["imagery_postfire"]["accepted"]]


def ybin(recs):
    return np.array([r["labels"]["outcome"] == "Destroyed" for r in recs], dtype=int)


def ylev(recs):
    return np.array([LEVELS.index(r["labels"]["damage"]) for r in recs])


def report(name, recs, p, extra=None):
    y, ev = ybin(recs), np.array([r["event"] for r in recs])
    ci = _cluster_boot(ev, lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan)
    r = {"model": name, "n": len(y), "auroc": float(roc_auc_score(y, p)), "auroc_ci95": ci,
         "balanced_acc": float(balanced_accuracy_score(y, p >= 0.5))}
    r.update(extra or {})
    print(r, flush=True)
    return r


def linear_probe(encoder, tr, va, te, prepost: bool):
    def feats(recs):
        post = embed([r["imagery_postfire"]["image_path"] for r in recs], encoder, "cuda")
        if not prepost:
            return post
        ok = [r["imagery"]["accepted"] for r in recs]
        pre = embed([r["imagery"]["image_path"] if o else r["imagery_postfire"]["image_path"]
                     for r, o in zip(recs, ok)], encoder, "cuda")
        return np.concatenate([pre, post, post - pre], axis=1)
    Xtr, Xte = feats(tr + va), feats(te)
    clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=128, random_state=0)),
                    ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=4000, scoring="roc_auc"))])
    clf.fit(Xtr, ybin(tr + va))
    return clf.predict_proba(Xte)[:, 1]


class DS(torch.utils.data.Dataset):
    def __init__(self, recs, proc, train, crop=True):
        self.recs, self.proc, self.train, self.crop = recs, proc, train, crop
        self.y = ylev(recs)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        im = Image.open(self.recs[i]["imagery_postfire"]["image_path"]).convert("RGB")
        if self.crop:
            im = im.crop((128, 128, 384, 384))
        if self.train:
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(90 * np.random.randint(4))
        x = self.proc(images=im, return_tensors="pt")["pixel_values"][0]
        return x, int(self.y[i])


def finetune(tr, va, te, epochs=8, lr=3e-5, bs=32, name="facebook/dinov2-small", crop=True):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name, num_labels=len(LEVELS)).cuda()
    counts = np.bincount(ylev(tr), minlength=len(LEVELS)) + 1
    w = torch.tensor((counts.sum() / counts) / (counts.sum() / counts).mean(), dtype=torch.float).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(DS(tr, proc, True, crop), batch_size=bs, shuffle=True, num_workers=4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl) + 5, pct_start=0.1)

    @torch.no_grad()
    def predict(recs):
        model.eval()
        out = []
        for x, _ in torch.utils.data.DataLoader(DS(recs, proc, False, crop), batch_size=64, num_workers=4):
            out.append(torch.softmax(model(pixel_values=x.cuda()).logits, -1).cpu().numpy())
        model.train()
        return np.concatenate(out)

    best, best_state = -1, None
    for ep in range(epochs):
        for x, y in dl:
            loss = torch.nn.functional.cross_entropy(model(pixel_values=x.cuda()).logits, y.cuda(), weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        pv = predict(va)
        auc = roc_auc_score(ybin(va), pv[:, 3:].sum(1))
        print(f"  epoch {ep}: val AUROC {auc:.4f}", flush=True)
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt = predict(te)
    return pt[:, 3:].sum(1), pt, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ft", action="store_true")
    ap.add_argument("--tuned-ft-only", action="store_true",
                    help="(review W8) tuned DINOv2-base fine-tune; config chosen on val among crop/full x lr")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = load("train"), load("val"), load("test")
    if args.tuned_ft_only:
        cands = []
        for crop in (True, False):
            for lr in (1e-5, 3e-5):
                p, probs, vbest = finetune(tr, va, te, epochs=12, lr=lr, bs=32, name="facebook/dinov2-base", crop=crop)
                cands.append({"crop": crop, "lr": lr, "val": vbest, "test_auroc": float(roc_auc_score(ybin(te), p)),
                              "test_macro_f1": float(f1_score(ylev(te), probs.argmax(1), average="macro")), "p": p})
                print({k: v for k, v in cands[-1].items() if k != "p"}, flush=True)
        best = max(cands, key=lambda c: c["val"])
        np.save(OUT / "ft_dinov2base_tuned_test_probs.npy", best["p"])
        (OUT / "ft_dinov2base_tuned.json").write_text(json.dumps(
            {"selected": {k: v for k, v in best.items() if k != "p"},
             "all": [{k: v for k, v in c.items() if k != "p"} for c in cands],
             "test_ids": [r["global_id"] for r in te]}, indent=2, default=float))
        print("SELECTED", {k: v for k, v in best.items() if k != "p"})
        return
    print(f"train {len(tr)} val {len(va)} test {len(te)}; test destroyed rate {ybin(te).mean():.2f}", flush=True)
    res = []
    preds = {"global_id": [r["global_id"] for r in te]}
    for enc in ["facebook/dinov2-base", "google/siglip-base-patch16-224"]:
        for tag, pp in (("lp_post", False), ("lp_prepost", True)):
            p = linear_probe(enc, tr, va, te, pp)
            preds[f"{tag} {enc}"] = p.tolist()
            res.append(report(f"{tag} {enc}", te, p))
    (OUT / "lp_test_preds.json").write_text(json.dumps(preds))
    if not args.skip_ft:
        p, probs, vbest = finetune(tr, va, te)
        res.append(report("ft_post dinov2-small (5-level head)", te, p, {
            "val_auroc_best": vbest,
            "macro_f1_5level": float(f1_score(ylev(te), probs.argmax(1), average="macro")),
            "test_pred_path": str(OUT / "ft_post_test_probs.npy")}))
        np.save(OUT / "ft_post_test_probs.npy", probs)
    (OUT / "supervised_baselines.json").write_text(json.dumps(res, indent=2, default=float))
    L = ["# Post-fire supervised baselines (fit: train events; select: val events; report: 2024+ test events)\n",
         "| Model | N test | AUROC [95% event CI] | Bal. acc | Notes |", "|---|---:|---|---:|---|"]
    for r in res:
        notes = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                 if k in ("val_auroc_best", "macro_f1_5level")}
        L.append(f"| {r['model']} | {r['n']} | {r['auroc']:.3f} [{r['auroc_ci95'][0]:.3f}, {r['auroc_ci95'][1]:.3f}] | "
                 f"{r['balanced_acc']:.3f} | {notes or ''} |")
    (OUT / "supervised_baselines.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
