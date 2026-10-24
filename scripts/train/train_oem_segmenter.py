"""
(V2) Land-cover segmenter for pre-fire "defensible space" perception, trained on OpenEarthMap (Xia et al., WACV 2023;
0.25-0.5 m, 8 classes). Applied later to our capture-date-verified Wayback crops (~0.47 m at z18) to measure
vegetation, bare ground, roads and neighbouring buildings in rings around each home.

Model: DINOv2-base backbone (facebook/dinov2-base); hidden states of blocks 3/6/9/12 are projected, fused and
upsampled by a small conv decoder to 8 classes (label 0 "unknown" is ignored). Backbone fine-tuned at a lower LR.
Train on the official train list, select by mIoU on the official val list (test labels are not used).

Data: run scripts/data/prepare_oem.py first (GeoTIFF -> PNG cache + split lists).
Output: output/models/oem_segmenter/{best.pt, log.json}
Run in the GPU env (needs LD_LIBRARY_PATH to the grpo env's nvidia/cu13/lib).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("/data/wildfire_benchmark_data/openearthmap/png")
OUT = ROOT / "output" / "models" / "oem_segmenter"
CLASSES = ["bareland", "rangeland", "developed", "road", "tree", "water", "agriculture", "building"]  # labels 1..8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Segmenter(nn.Module):
    def __init__(self, backbone="facebook/dinov2-base", layers=(3, 6, 9, 12), dim=256, n_cls=8):
        super().__init__()
        from transformers import Dinov2Model
        self.bb = Dinov2Model.from_pretrained(backbone)
        h = self.bb.config.hidden_size
        self.layers = layers
        self.proj = nn.ModuleList([nn.Conv2d(h, dim, 1) for _ in layers])
        self.fuse = nn.Sequential(nn.Conv2d(dim * len(layers), dim, 3, padding=1), nn.BatchNorm2d(dim), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.BatchNorm2d(dim // 2), nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2), nn.BatchNorm2d(dim // 4), nn.GELU(),
            nn.Conv2d(dim // 4, n_cls, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        out = self.bb(pixel_values=x, output_hidden_states=True)
        gh, gw = H // 14, W // 14
        feats = []
        for p, li in zip(self.proj, self.layers):
            t = out.hidden_states[li][:, 1:, :]  # drop CLS
            t = t.transpose(1, 2).reshape(t.shape[0], -1, gh, gw)
            feats.append(p(t))
        y = self.up(self.fuse(torch.cat(feats, 1)))
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class OEM(torch.utils.data.Dataset):
    def __init__(self, names, crop, train):
        self.names, self.crop, self.train = names, crop, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        im = np.array(Image.open(CACHE / "images" / f"{n}.png").convert("RGB"))
        lb = np.array(Image.open(CACHE / "labels" / f"{n}.png"))
        if self.train:
            h, w = lb.shape
            c = self.crop
            if h < c or w < c:
                im = np.pad(im, ((0, max(0, c - h)), (0, max(0, c - w)), (0, 0)))
                lb = np.pad(lb, ((0, max(0, c - h)), (0, max(0, c - w))))
                h, w = lb.shape
            y0, x0 = random.randint(0, h - c), random.randint(0, w - c)
            im, lb = im[y0:y0 + c, x0:x0 + c], lb[y0:y0 + c, x0:x0 + c]
            k = random.randint(0, 3)
            im, lb = np.rot90(im, k), np.rot90(lb, k)
            if random.random() < 0.5:
                im, lb = im[:, ::-1], lb[:, ::-1]
        im = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
        lb = torch.from_numpy(np.ascontiguousarray(lb)).long() - 1  # 0 (unknown) -> -1 (ignored)
        return im, lb


@torch.no_grad()
def evaluate(model, names, dev, tile=518):
    model.eval()
    inter = np.zeros(8)
    union = np.zeros(8)
    for n in names:
        im, lb = OEM([n], tile, False)[0]
        H, W = lb.shape
        ph, pw = (-H) % 14, (-W) % 14
        x = F.pad(im[None], (0, pw, 0, ph), mode="reflect").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(norm(x))
        pred = logits[0, :, :H, :W].argmax(0).cpu()
        valid = lb >= 0
        for c in range(8):
            p, g = (pred == c) & valid, (lb == c) & valid
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    iou = inter / np.maximum(union, 1)
    model.train()
    return float(iou.mean()), {c: round(float(v), 4) for c, v in zip(CLASSES, iou)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bb-lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--eval-every", type=int, default=2)
    a = ap.parse_args()
    random.seed(1019)
    torch.manual_seed(1019)
    split = json.loads((CACHE / "splits.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model = Segmenter().to(dev)
    params = [{"params": model.bb.parameters(), "lr": a.bb_lr},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("bb.")], "lr": a.lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    dl = torch.utils.data.DataLoader(OEM(split["train"], a.crop, True), batch_size=a.bs, shuffle=True,
                                     num_workers=8, drop_last=True, pin_memory=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.lr], total_steps=a.epochs * len(dl), pct_start=0.05)
    log = {"args": vars(a), "n_train": len(split["train"]), "n_val": len(split["val"]), "epochs": []}
    best = -1.0
    for ep in range(1, a.epochs + 1):
        t0, tot = time.time(), 0.0
        for im, lb in dl:
            im, lb = im.to(dev, non_blocking=True), lb.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(norm(im)).float(), lb, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(dl), "sec": round(time.time() - t0)}
        if ep % a.eval_every == 0 or ep == a.epochs:
            miou, per = evaluate(model, split["val"], dev)
            rec.update({"val_miou": miou, "val_iou": per})
            if miou > best:
                best = miou
                torch.save(model.state_dict(), OUT / "best.pt")
        log["epochs"].append(rec)
        log["best_val_miou"] = best
        (OUT / "log.json").write_text(json.dumps(log, indent=1))
        print(rec, flush=True)


if __name__ == "__main__":
    main()
