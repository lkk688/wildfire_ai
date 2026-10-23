"""
(Pilot B) Rule and learned baselines for evacuation replay items (output/evac/items.jsonl).

Scores (higher = evacuate sooner):
  distance        : -distance to the current perimeter
  wind_adjusted   : -distance / (1 + max(0, downwind_align) * wind_ms / 5)      (fixed, no fitting)
  logreg          : logistic regression on log distance, alignment, wind, gust, RH, log area, log growth,
                    fitted on 2022-23 fires (train + val), evaluated on 2024-25 test fires
Metrics per label: AUROC (fire-event bootstrap CI), mean within-decision-point AUROC, and the false-reassurance
rate (share of positives NOT alerted) when each method alerts the same fraction of items as are positive.
Output: output/evac/baselines.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

EVAC = ROOT / "output" / "evac"
LABELS = ["reached", "loss", "reached_24h"]


def feats(d):
    return np.c_[np.log1p(d["dist_m"]), d["downwind_align"], d["wind_ms"], d["gust_ms"], d["rh"],
                 d["downwind_align"] * d["wind_ms"], np.log1p(d["area_ha"]), np.log1p(d["growth_ha_h"].clip(lower=0))]


def within(d, s, lab):
    vals = []
    for _, g in d.assign(s=s).groupby("state_id"):
        if g[lab].nunique() == 2:
            vals.append(roc_auc_score(g[lab], g["s"]))
    return float(np.mean(vals)) if vals else None, len(vals)


def false_reassurance(y, s):
    k = int(y.sum())
    alerted = np.zeros(len(y), bool)
    alerted[np.argsort(-s, kind="stable")[:k]] = True
    return float(1 - (alerted & (y == 1)).sum() / max(k, 1))


def evaluate(d, s, lab):
    y = d[lab].to_numpy()
    ev = d["event"].to_numpy()
    w, nw = within(d, s, lab)
    return {"auroc": float(roc_auc_score(y, s)),
            "ci95": _cluster_boot(ev, lambda i: roc_auc_score(y[i], s[i]) if len(set(y[i])) == 2 else np.nan),
            "within_decision_auroc": w, "n_decisions": nw, "false_reassurance_at_matched_alerts": false_reassurance(y, s)}


def main():
    d = pd.DataFrame([json.loads(x) for x in (EVAC / "items.jsonl").read_text().splitlines()])
    tr, te = d[d["split"].isin(["train", "val"])], d[d["split"] == "test"].reset_index(drop=True)
    res = {"n_test": len(te), "n_train": len(tr), "test_events": int(te["event"].nunique()),
           "pos_rate_test": {lab: float(te[lab].mean()) for lab in LABELS}}
    full_te = te
    for band in (None, 5000, 2000):
        te = full_te if band is None else full_te[full_te["dist_m"] <= band].reset_index(drop=True)
        key = "all" if band is None else f"within_{band // 1000}km"
        res[key] = {"n": len(te), "pos_rate": {lab: float(te[lab].mean()) for lab in LABELS}}
        for lab in LABELS:
            res[key][lab] = run_label(tr, te, lab)
            print(key, lab, {k: (round(v["auroc"], 3), round(v["false_reassurance_at_matched_alerts"], 3))
                             for k, v in res[key][lab].items()}, flush=True)
    te = full_te
    print("per-event positives (all):\n", te.groupby("event")[LABELS].agg(["mean", "size"]).round(3).to_string())
    (EVAC / "baselines.json").write_text(json.dumps(res, indent=2, default=float))


def run_label(tr, te, lab):
    if True:
        sc = {"distance": -te["dist_m"].to_numpy(),
              "wind_adjusted": -(te["dist_m"] / (1 + te["downwind_align"].clip(lower=0) * te["wind_ms"] / 5)).to_numpy()}
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(feats(tr), tr[lab])
        sc["logreg"] = m.predict_proba(feats(te))[:, 1]
        return {k: evaluate(te, v, lab) for k, v in sc.items()}


if __name__ == "__main__":
    main()
