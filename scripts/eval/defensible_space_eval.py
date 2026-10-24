"""
(V2) Do defensible-space features seen from above predict loss once fire exposure is controlled?

Features: output/experiments/defensible_space/features.parquet (scripts/eval/defensible_space_features.py).
Metadata: structure type, year built, assessed value (known before a fire).

T1 (single buildings): logistic regression fitted on train_sample (2019-23 fires), evaluated on pilot_test
    (diagnostic set), test_natural_xl (natural prevalence) and confirm_direct (untouched val fires).
    Feature sets: meta | defensible space (DS) | meta + DS.  AUROC with fire-event bootstrap.
T2 (same-fire neighbours 15-100 m apart): logistic ranker on feature differences fitted on train_pairs, evaluated on
    the 403 main test pairs, the 628 XL pairs and the 289 replication pairs.  Also univariate: for each feature, the
    share of test pairs where the destroyed home has the higher value (0.5 = no difference), with a sign-test p.
All results also split into Eaton + Palisades vs other fires.
Output: output/experiments/defensible_space/eval.json and eval.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"
OUT = ROOT / "output" / "experiments" / "defensible_space"
URBAN = ("Eaton", "Palisades")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def meta(r):
    p = r["inputs"]["prefire"]
    yb = p.get("year_built")
    v = p.get("assessed_improved_value")
    st = str(p.get("structure_type") or "")
    return {"m_year_built": float(yb) if yb else np.nan, "m_log_value": float(np.log1p(v)) if v else np.nan,
            "m_mobile": float("Mobile" in st), "m_multi": float("Multi" in st)}


def main():
    F = pd.read_parquet(OUT / "features.parquet").set_index("global_id")
    ds_cols = [c for c in F.columns if c.startswith("r") and "_" in c] + ["nearest_building_m", "n_buildings_30m",
                                                                          "home_area_m2", "home_found"]
    ds_cols = [c for c in ds_cols if c in F.columns]
    allrec = {}
    for f in ["train_sample", "pilot_test", "test_natural_xl", "confirm_direct"]:
        for r in jl(SPL / f"{f}.jsonl"):
            allrec[r["global_id"]] = r
    for f in ["train_pairs", "pairs_test", "pairs_test_large", "pairs_test_xl", "confirm_pairs"]:
        for q in jl(SPL / f"{f}.jsonl"):
            for s in ("destroyed", "survived"):
                allrec[q[s]["global_id"]] = q[s]
    M = pd.DataFrame({g: meta(r) for g, r in allrec.items()}).T
    X = F.join(M, how="left")
    meta_cols = list(M.columns)
    sets = {"meta": meta_cols, "ds": ds_cols, "meta+ds": meta_cols + ds_cols}
    model = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                                  LogisticRegression(C=0.1, max_iter=4000))
    res = {"n_features_ds": len(ds_cols), "T1": {}, "T2": {}, "univariate": {}}

    # ---------------- T1
    tr = [r["global_id"] for r in jl(SPL / "train_sample.jsonl") if r["global_id"] in X.index]
    ytr = X.loc[tr, "y"].to_numpy()
    for test in ["pilot_test", "test_natural_xl", "confirm_direct"]:
        recs = [r for r in jl(SPL / f"{test}.jsonl") if r["global_id"] in X.index]
        if test == "pilot_test":
            recs = [r for r in recs if r.get("pilot_set") == "diagnostic"]
        ids = [r["global_id"] for r in recs]
        y = X.loc[ids, "y"].to_numpy()
        ev = np.array([r["event"] for r in recs])
        res["T1"][test] = {"n": len(ids)}
        for name, cols in sets.items():
            p = model().fit(X.loc[tr, cols], ytr).predict_proba(X.loc[ids, cols])[:, 1]
            out = {"auroc": float(roc_auc_score(y, p)),
                   "ci95": _cluster_boot(ev, lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan)}
            urb = np.array([e.startswith(URBAN) for e in ev])
            for tag, m in (("eaton_palisades", urb), ("other", ~urb)):
                if m.sum() > 20 and len(set(y[m])) == 2:
                    out[tag] = float(roc_auc_score(y[m], p[m]))
            res["T1"][test][name] = out

    # ---------------- T2
    def pairs(f):
        return [q for q in jl(SPL / f"{f}.jsonl") if q["destroyed"]["global_id"] in X.index and q["survived"]["global_id"] in X.index]

    trp = pairs("train_pairs")
    rng = np.random.default_rng(1019)

    def diff(ps, cols):
        A = X.loc[[q["destroyed"]["global_id"] for q in ps], cols].to_numpy(float)
        B = X.loc[[q["survived"]["global_id"] for q in ps], cols].to_numpy(float)
        return A, B

    for name, cols in sets.items():
        A, B = diff(trp, cols)
        med = np.nanmedian(np.vstack([A, B]), 0)
        A, B = np.where(np.isnan(A), med, A), np.where(np.isnan(B), med, B)
        sd = np.vstack([A, B]).std(0) + 1e-9
        flip = rng.random(len(A)) < 0.5
        D = np.where(flip[:, None], B - A, A - B) / sd
        rk = LogisticRegression(C=0.1, fit_intercept=False, max_iter=4000).fit(D, (~flip).astype(int))
        for test in ["pairs_test", "pairs_test_large", "pairs_test_xl", "confirm_pairs"]:
            ps = pairs(test)
            a_, b_ = diff(ps, cols)
            a_, b_ = np.where(np.isnan(a_), med, a_), np.where(np.isnan(b_), med, b_)
            s = rk.decision_function((a_ - b_) / sd)
            acc = (s > 0) + 0.5 * (s == 0)
            ev = np.array([q["event"] for q in ps])
            key = "main_403" if test in ("pairs_test", "pairs_test_large") else test
            res["T2"].setdefault(key, {})
            res["T2"][key].setdefault("_acc", {}).setdefault(name, []).append((acc, ev))
    for key, d in res["T2"].items():
        accs = d.pop("_acc")
        for name, parts in accs.items():
            acc = np.concatenate([p[0] for p in parts])
            ev = np.concatenate([p[1] for p in parts])
            urb = np.array([e.startswith(URBAN) for e in ev])
            d[name] = {"pair_acc": float(acc.mean()), "n": int(len(acc)),
                       "ci95": _cluster_boot(ev, lambda i: acc[i].mean()),
                       "eaton_palisades": float(acc[urb].mean()) if urb.any() else None,
                       "other": float(acc[~urb].mean()) if (~urb).any() else None}

    # ---------------- univariate (all test pairs pooled: main + XL)
    ps = pairs("pairs_test") + pairs("pairs_test_large") + pairs("pairs_test_xl")
    for c in ds_cols + meta_cols:
        a_, b_ = diff(ps, [c])
        ok = ~np.isnan(a_[:, 0]) & ~np.isnan(b_[:, 0]) & (a_[:, 0] != b_[:, 0])
        if ok.sum() < 30:
            continue
        k = int((a_[ok, 0] > b_[ok, 0]).sum())
        res["univariate"][c] = {"share_destroyed_higher": k / ok.sum(), "n": int(ok.sum()),
                                "p_sign": float(binomtest(k, int(ok.sum()), 0.5).pvalue)}
    (OUT / "eval.json").write_text(json.dumps(res, indent=2, default=float))

    L = ["# Defensible-space features (OpenEarthMap segmenter on verified pre-fire crops)\n", "## T1 AUROC\n",
         "| Test set | n | meta | DS | meta+DS |", "|---|---:|---|---|---|"]
    for t, d in res["T1"].items():
        L.append(f"| {t} | {d['n']} | " + " | ".join(f"{d[k]['auroc']:.3f} [{d[k]['ci95'][0]:.2f},{d[k]['ci95'][1]:.2f}]" for k in sets) + " |")
    L += ["\n## T2 pair accuracy (ranker on feature differences, fitted on train pairs)\n",
          "| Pair set | n | meta | DS | meta+DS | DS on Eaton+Palisades / other |", "|---|---:|---|---|---|---|"]
    for t, d in res["T2"].items():
        L.append(f"| {t} | {d['ds']['n']} | " + " | ".join(f"{d[k]['pair_acc']:.3f} [{d[k]['ci95'][0]:.2f},{d[k]['ci95'][1]:.2f}]" for k in sets)
                 + f" | {d['ds']['eaton_palisades'] if d['ds']['eaton_palisades'] is None else round(d['ds']['eaton_palisades'], 3)} / "
                   f"{d['ds']['other'] if d['ds']['other'] is None else round(d['ds']['other'], 3)} |")
    L += ["\n## Univariate: share of test pairs where the destroyed home has the higher value (p < 0.01 shown)\n",
          "| Feature | share | n | sign-test p |", "|---|---:|---:|---:|"]
    for c, d in sorted(res["univariate"].items(), key=lambda kv: kv[1]["p_sign"]):
        if d["p_sign"] < 0.01:
            L.append(f"| {c} | {d['share_destroyed_higher']:.3f} | {d['n']} | {d['p_sign']:.1e} |")
    (OUT / "eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
