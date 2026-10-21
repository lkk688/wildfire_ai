"""
Final comparison for direction 2 (docs/POSTFIRE_TASK_PLAN.md sections 5 / S1-S4).

All models are compared on the SAME buildings:
  subset400  the fixed stratified 400-building test subset the Claude models ran on
  full       all 1,246 test buildings with both crops (non-Claude models)
Paired differences use a fire-event cluster bootstrap.  Per-event AUROC for
Eaton and Palisades (S4).  The linear-probe rows are a stricter, post-hoc
reference (the pre-registered S2 reference is the fine-tuned DINOv2-small).

Output: output/experiments/postfire/comparison.{json,md}
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "output" / "experiments"
PF = EXP / "postfire"
SEED = 1019


def zero_shot(run, evidence):
    best = {}
    for line in (EXP / "vlm_conditions" / run / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] != evidence:
            continue
        if r["global_id"] not in best or r.get("parse_ok"):
            best[r["global_id"]] = r
    return {g: r["p_destroyed"] for g, r in best.items() if r.get("parse_ok")}


def main() -> None:
    te = [json.loads(line) for line in (ROOT / "output/splits/postfire_test.jsonl").read_text().splitlines()]
    te = {r["global_id"]: r for r in te if r["imagery_postfire"]["accepted"] and r["imagery"]["accepted"]}
    y = {g: int(r["labels"]["outcome"] == "Destroyed") for g, r in te.items()}
    ev = {g: r["event"] for g, r in te.items()}

    scores = {}
    sft = json.loads((PF / "sft_Qwen3.5-4B" / "log.json").read_text())
    pt = np.load(PF / "sft_Qwen3.5-4B" / "test_probs.npy")
    scores["Qwen3.5-4B + LoRA SFT (ours)"] = dict(zip(sft["test_ids"], pt[:, 3:].sum(1)))
    base = json.loads((PF / "sft_base_eval" / "log.json").read_text())
    pb = np.load(PF / "sft_base_eval" / "test_base_probs.npy")
    scores["Qwen3.5-4B base (same readout)"] = dict(zip(base["ids"], pb[:, 3:].sum(1)))
    for name, run in [("Qwen3.5-4B zero-shot (generative)", "pf_test_qwen35_4b_nothink"),
                      ("Qwen3.8-27B zero-shot", "pf_test_qwen38"),
                      ("Claude Sonnet 5.5 zero-shot", "pf_test_claude_sonnet_5_5"),
                      ("Claude Haiku 5.5 zero-shot", "pf_test_claude_haiku_5_5")]:
        scores[f"{name} | post"] = zero_shot(run, "postfire_image")
        scores[f"{name} | pre+post"] = zero_shot(run, "prepost_image")
    lp = json.loads((PF / "lp_test_preds.json").read_text())
    for k in ("lp_post facebook/dinov2-base", "lp_prepost facebook/dinov2-base"):
        scores[f"{k} (post-hoc stricter ref)"] = dict(zip(lp["global_id"], lp[k]))
    ft_ids = [json.loads(line)["global_id"] for line in (ROOT / "output/splits/postfire_test.jsonl").read_text().splitlines()
              if json.loads(line)["imagery_postfire"]["accepted"]]
    ft = np.load(PF / "ft_post_test_probs.npy")
    scores["ft DINOv2-small (pre-registered S2 ref)"] = dict(zip(ft_ids, ft[:, 3:].sum(1)))

    claude_ids = set(scores["Claude Haiku 5.5 zero-shot | post"]) & set(scores["Claude Sonnet 5.5 zero-shot | post"])
    sets = {"subset400": sorted(g for g in claude_ids if g in te),
            "full": sorted(te)}
    rng = np.random.default_rng(SEED)

    def auc(g, s):
        yy = np.array([y[i] for i in g])
        return roc_auc_score(yy, [s[i] for i in g]) if len(set(yy)) == 2 else np.nan

    def boot_diff(g, s1, s2, n=1000):
        evs = np.array([ev[i] for i in g])
        u = np.unique(evs)
        idx = {e: np.where(evs == e)[0] for e in u}
        ga = np.array(g)
        out = []
        for _ in range(n):
            pick = ga[np.concatenate([idx[e] for e in rng.choice(u, len(u))])]
            d = auc(pick, s1) - auc(pick, s2)
            if not np.isnan(d):
                out.append(d)
        return np.percentile(out, [2.5, 97.5]).tolist()

    res = {"rows": [], "paired": [], "per_event": []}
    L = ["# Post-fire damage recognition: final comparison (2024+ test events)\n"]
    for sname, g_all in sets.items():
        L += [f"\n## {sname} ({len(g_all)} buildings, destroyed rate {np.mean([y[i] for i in g_all]):.2f})\n",
              "| Model | N scored | AUROC [95% event CI] |", "|---|---:|---|"]
        for name, s in scores.items():
            g = [i for i in g_all if i in s]
            if len(g) < 0.95 * len(g_all):
                continue
            evs = np.array([ev[i] for i in g])
            u = np.unique(evs)
            idx = {e: np.where(evs == e)[0] for e in u}
            ga = np.array(g)
            bs = [auc(ga[np.concatenate([idx[e] for e in rng.choice(u, len(u))])], s) for _ in range(1000)]
            ci = np.nanpercentile(bs, [2.5, 97.5])
            a = auc(g, s)
            res["rows"].append({"set": sname, "model": name, "n": len(g), "auroc": a, "ci95": ci.tolist()})
            L.append(f"| {name} | {len(g)} | {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    ours = scores["Qwen3.5-4B + LoRA SFT (ours)"]
    L += ["\n## Paired differences (ours − reference, same buildings, event bootstrap)\n",
          "| Set | Reference | Δ AUROC | 95% CI |", "|---|---|---:|---|"]
    for sname, refs in [("subset400", ["Claude Haiku 5.5 zero-shot | post", "Claude Haiku 5.5 zero-shot | pre+post",
                                       "Claude Sonnet 5.5 zero-shot | post", "Claude Sonnet 5.5 zero-shot | pre+post"]),
                        ("full", ["Qwen3.5-4B base (same readout)", "Qwen3.8-27B zero-shot | post",
                                  "ft DINOv2-small (pre-registered S2 ref)", "lp_post facebook/dinov2-base (post-hoc stricter ref)",
                                  "lp_prepost facebook/dinov2-base (post-hoc stricter ref)"])]:
        for ref in refs:
            g = [i for i in sets[sname] if i in ours and i in scores[ref]]
            d = auc(g, ours) - auc(g, scores[ref])
            ci = boot_diff(g, ours, scores[ref])
            res["paired"].append({"set": sname, "reference": ref, "delta": d, "ci95": ci, "n": len(g)})
            L.append(f"| {sname} | {ref} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    L += ["\n## Per-event AUROC (full set, events with >= 20 of each class)\n"]
    evs_ok = [e for e in sorted(set(ev.values()))
              if sum(1 for g in te if ev[g] == e and y[g]) >= 20 and sum(1 for g in te if ev[g] == e and not y[g]) >= 20]
    names = ["Qwen3.5-4B + LoRA SFT (ours)", "Qwen3.5-4B base (same readout)", "Qwen3.8-27B zero-shot | post",
             "lp_post facebook/dinov2-base (post-hoc stricter ref)", "ft DINOv2-small (pre-registered S2 ref)"]
    L.append("| Event | N | " + " | ".join(n.split(" (")[0] for n in names) + " |")
    L.append("|---|---:|" + "---:|" * len(names))
    for e in evs_ok:
        g = [i for i in te if ev[i] == e]
        row = [auc([i for i in g if i in scores[n]], scores[n]) for n in names]
        res["per_event"].append({"event": e, "n": len(g), **dict(zip(names, row))})
        L.append(f"| {e.split(' |')[0]} | {len(g)} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    L.append(f"\nSFT 5-level macro-F1 (test): {sft['test']['macro_f1_5level']:.3f}; "
             f"base (same readout) {base['test_base']['macro_f1_5level']:.3f}; "
             f"ft DINOv2-small 0.276 (supervised_baselines_full.md). Selected epoch {sft['best_epoch']}, "
             f"best val AUROC {sft['best_val_auroc']:.3f}, stop rule triggered: {sft['stop_rule_triggered']}.")
    (PF / "comparison.json").write_text(json.dumps(res, indent=2, default=float))
    (PF / "comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
