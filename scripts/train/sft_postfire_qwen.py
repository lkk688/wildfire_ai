"""
LoRA SFT of a small Qwen3.5 VLM for post-fire damage recognition
(docs/POSTFIRE_TASK_PLAN.md, section 4 + amendment 5c).

Target (short, no chain of thought):  {"damage_level": "<level>", "p_destroyed": <int>}
with level in none | affected | minor | major | destroyed (DINS 5 levels).
Scoring: one forward pass; softmax of the FIRST level token over the 5 options;
P(destroyed) = P(major) + P(destroyed).  The base model (epoch 0, no LoRA) is
scored the same way on the same prompt, so pre/post-SFT is apples-to-apples.

Selection on val AUROC only; test is scored once with the selected adapter.
Stop rule: if best val AUROC < epoch-0 val AUROC + 0.03 -> report and stop.

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  /home/lkk/miniconda3/envs/grpo/bin/python scripts/train/sft_postfire_qwen.py --model Qwen/Qwen3.5-4B
Outputs: output/experiments/postfire/sft_<model>/ {log.json, val_epoch*.npy, test_probs.npy, adapter/}
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
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPL = ROOT / "output" / "splits"
LEVEL_MAP = {"No Damage": "none", "Affected (>0-10%)": "affected", "Minor (10-25%)": "minor",
             "Major (25-50%)": "major", "Destroyed (>50%)": "destroyed"}
LEVELS = ["none", "affected", "minor", "major", "destroyed"]
P_TARGET = {"none": 2, "affected": 8, "minor": 20, "major": 75, "destroyed": 98}
LORA_TARGETS = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"
PROMPT = (
    "You are assessing a residential building after a California wildfire.\n"
    "Attached: a POST-fire aerial orthophoto (about 0.3-0.6 m/pixel, 512x512 px, about 150 m across). "
    "The building being assessed is at the exact centre of the image.\n"
    "Classify its damage using the CAL FIRE inspection levels: none (no damage), affected (1-10%), "
    "minor (10-25%), major (25-50%), destroyed (>50%).\n"
    'Reply with JSON only: {"damage_level": "<none|affected|minor|major|destroyed>", "p_destroyed": <0-100>}'
)
PREFIX = '{"damage_level": "'


def load(split):
    recs = [json.loads(line) for line in (SPL / f"postfire_{split}.jsonl").read_text().splitlines()]
    return [r for r in recs if r["imagery_postfire"]["accepted"] and r["imagery"]["accepted"]]


def target_text(rec):
    lv = LEVEL_MAP[rec["labels"]["damage"]]
    return PREFIX + lv + f'", "p_destroyed": {P_TARGET[lv]}}}'


class Batcher:
    def __init__(self, processor):
        self.p = processor
        tok = processor.tokenizer
        self.level_ids = []
        for lv in LEVELS:
            ids = tok(PREFIX + lv, add_special_tokens=False)["input_ids"]
            base = tok(PREFIX, add_special_tokens=False)["input_ids"]
            assert ids[:len(base)] == base, "prefix tokenisation is not stable"
            self.level_ids.append(ids[len(base)])
        assert len(set(self.level_ids)) == 5, f"level first tokens collide: {self.level_ids}"

    def _msgs(self, rec):
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]

    def _chat(self, rec):
        return self.p.apply_chat_template(self._msgs(rec), add_generation_prompt=True, tokenize=False,
                                          enable_thinking=False)

    def train_batch(self, recs, augment=True):
        ims, texts, prompt_lens = [], [], []
        for r in recs:
            im = Image.open(r["imagery_postfire"]["image_path"]).convert("RGB")
            if augment:
                if random.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                im = im.rotate(90 * random.randint(0, 3))
            prompt = self._chat(r)
            ims.append(im)
            texts.append(prompt + target_text(r) + self.p.tokenizer.eos_token)
            prompt_lens.append(prompt)
        enc = self.p(text=texts, images=ims, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        for i, pr in enumerate(prompt_lens):
            n_prompt = self.p(text=[pr], images=[ims[i]], return_tensors="pt")["input_ids"].shape[1]
            offset = 0 if self.p.tokenizer.padding_side == "right" else int((enc["attention_mask"][i] == 0).sum())
            labels[i, : offset + n_prompt] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    def score_batch(self, recs):
        ims = [Image.open(r["imagery_postfire"]["image_path"]).convert("RGB") for r in recs]
        texts = [self._chat(r) + PREFIX for r in recs]
        self.p.tokenizer.padding_side = "left"
        enc = self.p(text=texts, images=ims, return_tensors="pt", padding=True)
        self.p.tokenizer.padding_side = "right"
        return enc


@torch.no_grad()
def score(model, batcher, recs, bs=8):
    model.eval()
    out = []
    for i in range(0, len(recs), bs):
        enc = batcher.score_batch(recs[i:i + bs]).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        out.append(torch.softmax(logits[:, batcher.level_ids], -1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def metrics(recs, probs):
    y = np.array([r["labels"]["outcome"] == "Destroyed" for r in recs], dtype=int)
    lv = np.array([LEVELS.index(LEVEL_MAP[r["labels"]["damage"]]) for r in recs])
    p = probs[:, 3:].sum(1)
    return {"auroc": float(roc_auc_score(y, p)), "macro_f1_5level": float(f1_score(lv, probs.argmax(1), average="macro")),
            "n": len(recs)}


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-only-base", action="store_true", help="only score the base model (epoch 0)")
    ap.add_argument("--also-test-base", action="store_true",
                    help="also score the untrained base on test (saved separately; never used for selection)")
    args = ap.parse_args()
    random.seed(1019)
    torch.manual_seed(1019)
    out = ROOT / "output" / "experiments" / "postfire" / f"sft_{args.model.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = load("train"), load("val"), load("test")
    print(f"train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    batcher = Batcher(proc)
    log = {"args": vars(args), "epochs": []}

    t0 = time.time()
    pv0 = score(model, batcher, va)
    m0 = metrics(va, pv0)
    np.save(out / "val_epoch0.npy", pv0)
    log["epochs"].append({"epoch": 0, "val": m0, "sec": time.time() - t0})
    print("epoch 0 (base, no LoRA):", m0, flush=True)
    if args.also_test_base and not args.eval_only_base:
        ptb = score(model, batcher, te)
        np.save(out / "test_base_probs.npy", ptb)
        log["test_base"] = metrics(te, ptb)
        print("test base (no LoRA, same readout):", log["test_base"], flush=True)
    if args.eval_only_base:
        pt0 = score(model, batcher, te)
        np.save(out / "test_base_probs.npy", pt0)
        log["test_base"] = metrics(te, pt0)
        (out / "log.json").write_text(json.dumps(log, indent=2))
        print("test base:", log["test_base"])
        return

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = len(tr) // (args.bs * args.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs + 5,
                                                pct_start=0.05, anneal_strategy="cos")
    best, best_ep = m0["auroc"], 0
    for ep in range(1, args.epochs + 1):
        order = list(range(len(tr)))
        random.shuffle(order)
        tl, n = 0.0, 0
        for bi in range(0, len(order) - args.bs + 1, args.bs):
            enc = batcher.train_batch([tr[j] for j in order[bi:bi + args.bs]]).to(model.device)
            loss = model(**enc).loss / args.grad_accum
            loss.backward()
            tl += loss.item() * args.grad_accum
            n += 1
            if n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if n % 100 == 0:
                print(f"  ep{ep} batch {n} loss {tl / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        pv = score(model, batcher, va)
        m = metrics(va, pv)
        np.save(out / f"val_epoch{ep}.npy", pv)
        log["epochs"].append({"epoch": ep, "train_loss": tl / max(n, 1), "val": m, "sec": time.time() - t0})
        print(f"epoch {ep}:", m, flush=True)
        if m["auroc"] > best:
            best, best_ep = m["auroc"], ep
            model.save_pretrained(out / "adapter")
        (out / "log.json").write_text(json.dumps(log, indent=2))

    log["best_epoch"], log["best_val_auroc"] = best_ep, best
    log["stop_rule_triggered"] = best < m0["auroc"] + 0.03
    if best_ep > 0:
        from peft import PeftModel
        base = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
        model = PeftModel.from_pretrained(base, out / "adapter")
    pt = score(model, batcher, te)
    np.save(out / "test_probs.npy", pt)
    log["test"] = metrics(te, pt)
    log["test_ids"] = [r["global_id"] for r in te]
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print("TEST (selected epoch %d):" % best_ep, log["test"], flush=True)


if __name__ == "__main__":
    main()
