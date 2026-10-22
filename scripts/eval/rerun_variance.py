"""
Run-to-run variance of closed models (no fixed temperature): compare two identical runs.
Output: output/experiments/rerun_variance.{json,md}
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "output" / "experiments" / "vlm_conditions"


def load(run, evidence):
    best = {}
    for line in (EXP / run / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] == evidence and r["context"] == "neutral" and (r["global_id"] not in best or r.get("parse_ok")):
            best[r["global_id"]] = r
    return {g: r for g, r in best.items() if r.get("parse_ok")}


rows = []
for task, ev, pairs in [("pre-fire T1 (image)", "image", [("Sonnet", "pilot_sonnet_diag", "rep2_pre_claude_sonnet_5_5"),
                                                        ("Haiku", "pilot_haiku_diag", "rep2_pre_claude_haiku_5_5")]),
                        ("post-fire T3 subset-400", "postfire_image", [("Sonnet", "pf_test_claude_sonnet_5_5", "rep2_pf_claude_sonnet_5_5"),
                                                                     ("Haiku", "pf_test_claude_haiku_5_5", "rep2_pf_claude_haiku_5_5")])]:
    for m, r1, r2 in pairs:
        a, b = load(r1, ev), load(r2, ev)
        ids = sorted(set(a) & set(b))
        y = np.array([a[g]["label"] == "Destroyed" for g in ids], dtype=int)
        p1 = np.array([a[g]["p_destroyed"] for g in ids])
        p2 = np.array([b[g]["p_destroyed"] for g in ids])
        rows.append({"task": task, "model": m, "n": len(ids), "auroc_run1": roc_auc_score(y, p1),
                     "auroc_run2": roc_auc_score(y, p2), "auroc_mean": roc_auc_score(y, (p1 + p2) / 2),
                     "pearson_p": float(np.corrcoef(p1, p2)[0, 1]),
                     "label_agreement": float(np.mean((p1 >= 0.5) == (p2 >= 0.5))),
                     "mean_abs_dp": float(np.mean(np.abs(p1 - p2)))})
L = ["# Run-to-run variance (two identical runs, API-default sampling)\n",
     "| Task | Model | N | AUROC run 1 | AUROC run 2 | |Δ| | AUROC of averaged p | corr(p1,p2) | label agreement | mean |Δp| |",
     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    L.append(f"| {r['task']} | {r['model']} | {r['n']} | {r['auroc_run1']:.3f} | {r['auroc_run2']:.3f} | "
             f"{abs(r['auroc_run1'] - r['auroc_run2']):.3f} | {r['auroc_mean']:.3f} | {r['pearson_p']:.2f} | "
             f"{r['label_agreement']:.2f} | {r['mean_abs_dp']:.3f} |")
(ROOT / "output" / "experiments" / "rerun_variance.json").write_text(json.dumps(rows, indent=2))
(ROOT / "output" / "experiments" / "rerun_variance.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
