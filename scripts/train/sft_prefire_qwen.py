"""
(Review W2) Pre-fire LoRA: the symmetric counterpart of the post-fire SFT.
Same recipe (Qwen3.5-4B, LoRA r=16 on the language model, vision frozen, <=3 epochs,
first-token level readout) but the input is the time-verified PRE-fire crop and the
target is the damage level the building later suffered.

Train: all 2019-23 train-fire buildings with a pre-fire crop that we have
       (train_sample.jsonl + both sides of train_pairs.jsonl; deduplicated).
Select: epoch by AUROC on val-fire buildings (postfire_val.jsonl pre-fire crops).
Score once: T1 (pilot diagnostic, 238), T2 test pairs (403), replication pairs (289).
Output: output/experiments/prefire_sft_<model>/log.json (+ per-building probabilities)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.sft_postfire_qwen import LEVEL_MAP, LEVELS, LORA_TARGETS, P_TARGET, PREFIX  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
PROMPT = (
    "You are assessing a residential building in a California wildland-urban interface community, BEFORE a wildfire.\n"
    "Attached: a pre-fire aerial orthophoto (about 0.3-0.6 m/pixel, 512x512 px, about 150 m across). "
    "The building being assessed is at the exact centre of the image. A wildfire later reached this community.\n"
    "Predict the damage this building suffered, using the CAL FIRE inspection levels: none (no damage), affected (1-10%), "
    "minor (10-25%), major (25-50%), destroyed (>50%).\n"
    'Reply with JSON only: {"damage_level": "<none|affected|minor|major|destroyed>", "p_destroyed": <0-100>}'
)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


class Batcher:
    def __init__(self, processor):
        self.p = processor
        tok = processor.tokenizer
        base = tok(PREFIX, add_special_tokens=False)["input_ids"]
        self.level_ids = [tok(PREFIX + lv, add_special_tokens=False)["input_ids"][len(base)] for lv in LEVELS]
        assert len(set(self.level_ids)) == 5

    def _chat(self):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        return self.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def train_batch(self, recs):
        ims, texts = [], []
        for r in recs:
            im = Image.open(r["imagery"]["image_path"]).convert("RGB")
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(90 * random.randint(0, 3))
            lv = LEVEL_MAP[r["labels"]["damage"]]
            ims.append(im)
            texts.append(self._chat() + PREFIX + lv + f'", "p_destroyed": {P_TARGET[lv]}}}' + self.p.tokenizer.eos_token)
        enc = self.p(text=texts, images=ims, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        n_prompt = self.p(text=[self._chat()], images=[ims[0]], return_tensors="pt")["input_ids"].shape[1]
        labels[:, :n_prompt] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    def score_batch(self, recs):
        ims = [Image.open(r["imagery"]["image_path"]).convert("RGB") for r in recs]
        self.p.tokenizer.padding_side = "left"
        enc = self.p(text=[self._chat() + PREFIX] * len(recs), images=ims, return_tensors="pt", padding=True)
        self.p.tokenizer.padding_side = "right"
        return enc


@torch.no_grad()
def score(model, b, recs, bs=8):
    model.eval()
    out = []
    for i in range(0, len(recs), bs):
        enc = b.score_batch(recs[i:i + bs]).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        out.append(torch.softmax(logits[:, b.level_ids], -1)[:, 3:].sum(-1).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.array([])


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()
    random.seed(1019)
    torch.manual_seed(1019)
    out = ROOT / "output" / "experiments" / f"prefire_sft_{args.model.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)

    tr = {r["global_id"]: r for r in jl(SPL / "train_sample.jsonl") if r["imagery"]["accepted"]}
    for q in jl(SPL / "train_pairs.jsonl"):
        for s in ("destroyed", "survived"):
            tr[q[s]["global_id"]] = q[s]
    tr = list(tr.values())
    va = [r for r in jl(SPL / "postfire_val.jsonl") if r["imagery"]["accepted"]]
    t1 = [r for r in jl(SPL / "pilot_test.jsonl") if r["imagery"]["accepted"] and r["pilot_set"] == "diagnostic"]
    psets = {"test": jl(SPL / "pairs_test.jsonl") + jl(SPL / "pairs_test_large.jsonl"), "val": jl(SPL / "confirm_pairs.jsonl")}
    y = lambda rs: np.array([r["labels"]["outcome"] == "Destroyed" for r in rs], dtype=int)  # noqa: E731
    print(f"train {len(tr)} (destroyed {y(tr).mean():.2f}) val {len(va)} T1 {len(t1)}", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    b = Batcher(proc)

    def evaluate(m, final=False):
        r = {"val_auroc": float(roc_auc_score(y(va), score(m, b, va)))}
        if final:
            p1 = score(m, b, t1)
            ev1 = np.array([x["event"] for x in t1])
            y1 = y(t1)
            r["T1_auroc"] = float(roc_auc_score(y1, p1))
            r["T1_ci"] = _cluster_boot(ev1, lambda i: roc_auc_score(y1[i], p1[i]) if len(set(y1[i])) == 2 else np.nan)
            for sname, ps in psets.items():
                pd_ = score(m, b, [q["destroyed"] for q in ps])
                ps_ = score(m, b, [q["survived"] for q in ps])
                acc = (pd_ > ps_) + 0.5 * (pd_ == ps_)
                evp = np.array([q["event"] for q in ps])
                r[f"T2_{sname}_pair_acc"] = float(acc.mean())
                r[f"T2_{sname}_ci"] = _cluster_boot(evp, lambda i: acc[i].mean())
        return r

    log = {"args": vars(args), "n_train": len(tr), "epochs": []}
    base = evaluate(model, final=True)
    log["base"] = base
    print("base (no LoRA):", base, flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = len(tr) // (args.bs * args.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps * args.epochs + 5, pct_start=0.05)
    best, best_ep, t0 = base["val_auroc"], 0, time.time()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(tr)))
        random.shuffle(order)
        n = 0
        for bi in range(0, len(order) - args.bs + 1, args.bs):
            loss = model(**b.train_batch([tr[j] for j in order[bi:bi + args.bs]]).to(model.device)).loss / args.grad_accum
            loss.backward()
            n += 1
            if n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        r = evaluate(model)
        log["epochs"].append({"epoch": ep, **r, "sec": time.time() - t0})
        print(f"epoch {ep}:", r, flush=True)
        if r["val_auroc"] > best:
            best, best_ep = r["val_auroc"], ep
            model.save_pretrained(out / "adapter")
        (out / "log.json").write_text(json.dumps(log, indent=2, default=float))
    log["best_epoch"] = best_ep
    if best_ep > 0:
        from peft import PeftModel
        m = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).cuda(),
                                      out / "adapter")
        log["final"] = evaluate(m, final=True)
    else:
        log["final"] = base
    (out / "log.json").write_text(json.dumps(log, indent=2, default=float))
    print("FINAL (epoch %d):" % best_ep, log["final"], flush=True)


if __name__ == "__main__":
    main()
