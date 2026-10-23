"""
(Pilot A) Evaluate community-scale risk scores on 2024-25 test-fire cells.

Label: frac_destroyed (share of inspected homes destroyed); high_loss = frac >= 0.5.
Scores:
  tabular_hgb        assessor aggregates known before the fire (median year built, median log value, mobile-home share),
                     fitted on 2019-23 cells
  wc_veg_1km         share of tree/shrub/grass within 1 km (ESA WorldCover 2021, pre-fire for these fires), no fitting
  wc_neg_dist        minus distance from the homes to a contiguous wildland patch >= 50 ha, no fitting
  wc_logreg_lofo     logistic regression on the WorldCover features, cross-fitted leaving one fire out (an estimate of
                     what such a model could do, not a deployable model)
  dinov2_probe       frozen DINOv2 linear probe on the same z16 crops, fitted on 2019-23 cells (if embeddings exist)
  vlm:<run>          p_high_loss from output/community/vlm/<run>/raw.jsonl (unparsed -> 0.5)
Metrics: AUROC(high_loss) with fire-event bootstrap, Spearman with frac_destroyed, within-fire pair accuracy
(pairs in the same fire whose loss fractions differ by >= 0.3), on the cells every score covers.
Fact check: VLM distance-to-wildland and position vs WorldCover.
Output: output/community/eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

OUT = ROOT / "output" / "community"
WC = ["veg_frac_cell", "veg_frac_1km", "built_frac_1km", "dist_wildland_m"]
TAB = ["yb_median", "log_value_median", "mobile_frac"]


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []


def pair_acc(d, s):
    acc = []
    for _, g in d.assign(s=s).groupby("event"):
        f, sc = g["frac_destroyed"].to_numpy(), g["s"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if abs(f[i] - f[j]) >= 0.3:
                    hi, lo = (i, j) if f[i] > f[j] else (j, i)
                    acc.append(1.0 if sc[hi] > sc[lo] else 0.5 if sc[hi] == sc[lo] else 0.0)
    return float(np.mean(acc)) if acc else None, len(acc)


def metrics(d, s):
    y = d["high_loss"].to_numpy()
    ev = d["event"].to_numpy()
    pa, npairs = pair_acc(d, s)
    return {"auroc": float(roc_auc_score(y, s)),
            "ci95": _cluster_boot(ev, lambda i: roc_auc_score(y[i], s[i]) if len(set(y[i])) == 2 else np.nan),
            "spearman": float(spearmanr(s, d["frac_destroyed"]).correlation), "within_fire_pair_acc": pa, "n_pairs": npairs}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="comma-separated VLM runs to include (default: all)")
    a = ap.parse_args()
    cells = pd.DataFrame(jl(OUT / "cells.jsonl"))
    cells = cells[cells["imagery"].apply(lambda m: m["accepted"])].reset_index(drop=True)
    wc = pd.DataFrame(jl(OUT / "worldcover.jsonl"))
    if not wc.empty:
        cells = cells.merge(wc[["cell_id", "valid"] + WC], on="cell_id", how="left")
    tr = cells[cells["split"].isin(["train", "val"])]
    te = cells[cells["split"] == "test"].copy()
    scores = {}
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    hgb.fit(tr[TAB], tr["high_loss"])
    scores["tabular_hgb"] = pd.Series(hgb.predict_proba(te[TAB])[:, 1], index=te.index)
    if "veg_frac_1km" in te:
        ok = te["veg_frac_1km"].notna()
        scores["wc_veg_1km"] = te.loc[ok, "veg_frac_1km"]
        scores["wc_neg_dist"] = -te.loc[ok, "dist_wildland_m"].fillna(1e5)
        lofo = pd.Series(np.nan, index=te.index)
        X = te[WC].fillna({"dist_wildland_m": 1e4}).assign(dist_wildland_m=lambda x: np.log1p(x["dist_wildland_m"]))
        for e in te["event"].unique():
            m_tr, m_te = ok & (te["event"] != e), ok & (te["event"] == e)
            if m_te.any() and te.loc[m_tr, "high_loss"].nunique() == 2:
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[m_tr], te.loc[m_tr, "high_loss"])
                lofo[m_te] = lr.predict_proba(X[m_te])[:, 1]
        scores["wc_logreg_lofo"] = lofo.dropna()
    emb = OUT / "dinov2_probe_test.json"
    if emb.exists():
        p = json.loads(emb.read_text())
        scores["dinov2_probe"] = te["cell_id"].map(p).dropna()
    vlm_rows = {}
    for run_dir in sorted((OUT / "vlm").glob("*")):
        if a.runs and run_dir.name not in a.runs.split(","):
            continue
        best = {}
        for r in jl(run_dir / "raw.jsonl"):
            if r["cell_id"] not in best or r.get("parsed"):
                best[r["cell_id"]] = r
        vlm_rows[run_dir.name] = best
        scores[f"vlm:{run_dir.name}"] = te["cell_id"].map(
            lambda c: (best[c]["parsed"]["p_high_loss"] if best[c].get("parsed") else 0.5) if c in best else np.nan).dropna()
    common = te.index
    for s in scores.values():
        common = common.intersection(s.index)
    d = te.loc[common]
    res = {"n_test_cells": int(len(te)), "n_common": int(len(d)), "events": int(d["event"].nunique()),
           "high_loss_rate": float(d["high_loss"].mean()), "scores": {}}
    for k, s in scores.items():
        res["scores"][k] = metrics(d, s.loc[common].to_numpy())
        print(f"{k:22s} AUROC {res['scores'][k]['auroc']:.3f} {np.round(res['scores'][k]['ci95'], 3)}  "
              f"rho {res['scores'][k]['spearman']:.3f}  pair {res['scores'][k]['within_fire_pair_acc']:.3f} "
              f"({res['scores'][k]['n_pairs']})", flush=True)
    # stratified: the two 2025 urban conflagrations vs the other (WUI) fires
    res["by_fire_type"] = {}
    urban = d["event"].str.startswith(("Eaton", "Palisades")).to_numpy()
    for name, m in (("eaton_palisades", urban), ("other_fires", ~urban)):
        sub = d[m]
        if sub["high_loss"].nunique() < 2:
            continue
        res["by_fire_type"][name] = {"n": int(m.sum()), "high_loss_rate": float(sub["high_loss"].mean())}
        for k, s_ in scores.items():
            v = s_.loc[common].to_numpy()[m]
            res["by_fire_type"][name][k] = metrics(sub, v)
            print(f"  [{name}] {k:20s} AUROC {res['by_fire_type'][name][k]['auroc']:.3f}  "
                  f"pair {res['by_fire_type'][name][k]['within_fire_pair_acc']:.3f}", flush=True)
    # fact check of the VLM's intermediate claims against WorldCover
    fc = {}
    for run, best in vlm_rows.items():
        x = d.assign(vd=d["cell_id"].map(lambda c: (best.get(c, {}).get("parsed") or {}).get("dist_to_wildland_m")),
                     pos=d["cell_id"].map(lambda c: (best.get(c, {}).get("parsed") or {}).get("position")))
        x = x.dropna(subset=["vd", "dist_wildland_m"])
        if len(x) > 10:
            emb_wc = x["dist_wildland_m"] <= 10
            fc[run] = {"n": int(len(x)), "spearman_dist_vs_worldcover": float(spearmanr(x["vd"], x["dist_wildland_m"]).correlation),
                       "embedded_agreement": float(((x["pos"] == "embedded_in_wildland") == emb_wc).mean()),
                       "worldcover_embedded_rate": float(emb_wc.mean())}
    res["fact_check"] = fc
    print(json.dumps(fc, indent=1))
    (OUT / "eval.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
