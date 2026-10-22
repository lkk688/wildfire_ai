"""
(Review W6) Post-fire T3 robustness: AUROC by capture lag (days from fire start to the
post-fire capture) and with a strict label (Destroyed >50% vs No Damage only).
Models: SFT Qwen3.5-4B / 9B, frozen DINOv2 LP (main config), Qwen3.8-27B zero-shot (full test).
Output: output/experiments/postfire/lag_levels.md
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "output" / "experiments"
SPL = ROOT / "output" / "splits"


def main():
    te = {json.loads(l)["global_id"]: json.loads(l) for l in (SPL / "postfire_test.jsonl").read_text().splitlines()}
    sc = {}
    for m, d in [("SFT 4B", "sft_Qwen3.5-4B"), ("SFT 9B", "sft_Qwen3.5-9B")]:
        log = json.loads((EXP / "postfire" / d / "log.json").read_text())
        sc[m] = dict(zip(log["test_ids"], np.load(EXP / "postfire" / d / "test_probs.npy")[:, 3:].sum(1)))
    lp = json.loads((EXP / "postfire" / "lp_test_preds.json").read_text())
    sc["DINOv2 LP"] = dict(zip(lp["global_id"], lp["lp_post facebook/dinov2-base"]))
    best = {}
    for line in (EXP / "vlm_conditions" / "pf_test_qwen38" / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] == "postfire_image" and r.get("parse_ok"):
            best[r["global_id"]] = r["p_destroyed"]
    sc["Qwen3.8-27B zero-shot"] = best
    ids = [g for g in sc["SFT 4B"] if all(g in s for s in sc.values())]
    lag = {g: -te[g]["imagery_postfire"]["staleness_days"] for g in ids}
    dmg = {g: te[g]["labels"]["damage"] for g in ids}
    y = {g: int(te[g]["labels"]["outcome"] == "Destroyed") for g in ids}
    strata = [("all", lambda g: True), ("lag <= 60 d", lambda g: lag[g] <= 60), ("lag 61-180 d", lambda g: 60 < lag[g] <= 180),
              ("lag > 180 d", lambda g: lag[g] > 180)]
    L = ["# Post-fire T3 by capture lag and strict labels\n",
         f"Common test buildings: {len(ids)}; median lag {int(np.median(list(lag.values())))} d.\n",
         "| Stratum | N | destroyed | " + " | ".join(sc) + " |", "|---|---:|---:|" + "---:|" * len(sc)]
    for name, f in strata:
        g = [i for i in ids if f(i)]
        yy = [y[i] for i in g]
        if len(set(yy)) < 2:
            continue
        L.append(f"| {name} | {len(g)} | {sum(yy)} | " + " | ".join(f"{roc_auc_score(yy, [s[i] for i in g]):.3f}" for s in sc.values()) + " |")
    g = [i for i in ids if dmg[i] in ("Destroyed (>50%)", "No Damage")]
    yy = [int(dmg[i] == "Destroyed (>50%)") for i in g]
    L.append(f"| strict: Destroyed vs No Damage | {len(g)} | {sum(yy)} | " + " | ".join(f"{roc_auc_score(yy, [s[i] for i in g]):.3f}" for s in sc.values()) + " |")
    g = [i for i in ids if dmg[i] in ("Major (25-50%)", "No Damage")]
    yy = [int(dmg[i] == "Major (25-50%)") for i in g]
    L.append(f"| Major vs No Damage | {len(g)} | {sum(yy)} | " + " | ".join(f"{roc_auc_score(yy, [s[i] for i in g]):.3f}" for s in sc.values()) + " |")
    (EXP / "postfire" / "lag_levels.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
