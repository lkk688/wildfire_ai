"""
Analyse VLM-as-sensor extraction runs (run_vlm_extraction.py).

1. Verifiability: each field is checked against an *independent* source, never
   against the fire outcome.
     roof_material         DINS ROOFCONSTRUCTION, survived homes only (inspected standing)
     attached_deck...      DINS deck elevated / patio cover, survived homes only
     nearest_other_building / buildings_within_50m
                           DINS-point geometry (output/features/geometry.parquet)
     all fields            rotation invariance (orig vs rot90), when rot90 was run
   Reported: coverage (share not "cannot_determine"), accuracy on answered,
   and whether self-reported confidence separates right from wrong.
2. Decision head (VLM-as-sensor): one-hot extracted fields -> logistic regression
   fitted on the TRAIN sample, scored on the pilot test set and the pair set;
   compared with the same head on pre-fire metadata, and both combined.

Usage:
  python scripts/eval/analyze_extraction.py --run extract_qwen38
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_vlm_extraction import CD, SCHEMA  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

EXT = ROOT / "output" / "experiments" / "vlm_extraction"
SPL = ROOT / "output" / "splits"
GEOM = ROOT / "output" / "features" / "geometry.parquet"

ROOF_MAP = {"Asphalt": "asphalt_shingle", "Tile": "clay_concrete_tile", "Metal": "metal", "Wood": "wood_shake",
            "Concrete": "clay_concrete_tile"}  # 'Fire Resistant', 'Combustible', 'Other' are not material classes


def nn_bin(d):
    if pd.isna(d):
        return None
    return "lt5m" if d < 5 else "5_10m" if d < 10 else "10_20m" if d < 20 else "20_50m" if d < 50 else "gt50m"


def n50_bin(n):
    return "0" if n == 0 else "1_2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "gt10"


def load_records() -> dict:
    recs = {}
    for f in ["pilot_test.jsonl", "train_sample.jsonl"]:
        p = SPL / f
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                recs[r["global_id"]] = r
    for pp in sorted(SPL.glob("pairs_test*.jsonl")):
        for line in pp.read_text().splitlines():
            q = json.loads(line)
            for side in ("destroyed", "survived"):
                q[side]["split"] = "pairs"
                recs[q[side]["global_id"]] = q[side]
    return recs


def load_run(runs) -> pd.DataFrame:
    rows = {}
    runs = [runs] if isinstance(runs, str) else runs
    for line in (ln for run in runs for ln in (EXT / run / "raw.jsonl").read_text().splitlines()):
        r = json.loads(line)
        k = (r["global_id"], r["variant"])
        if r.get("parse_ok") or k not in rows:
            rows[k] = r
    out = []
    for (gid, var), r in rows.items():
        if not r.get("parse_ok"):
            continue
        d = {"global_id": gid, "variant": var}
        for k in SCHEMA:
            a = r["attrs"].get(k) or {}
            d[k] = a.get("v")
            d[k + "__c"] = a.get("c")
        out.append(d)
    return pd.DataFrame(out)


def verification(ext: pd.DataFrame, recs: dict) -> list:
    o = ext[ext["variant"] == "orig"].copy()
    o["label"] = [recs[g]["labels"]["outcome"] for g in o["global_id"]]
    insp = {g: recs[g]["inputs"]["inspector_postfire"] for g in o["global_id"]}
    geom = pd.read_parquet(GEOM).set_index("GLOBALID")
    res = []

    def report(name, pred, truth, conf):
        pred, truth, conf = np.array(pred, dtype=object), np.array(truth, dtype=object), np.array(conf, dtype=float)
        has_truth = np.array([t is not None for t in truth])
        answered = has_truth & np.array([p not in (None, CD) for p in pred])
        n = int(has_truth.sum())
        if n == 0:
            return
        correct = np.array([p == t for p, t in zip(pred, truth)])
        acc = float(correct[answered].mean()) if answered.any() else float("nan")
        # does confidence rank correct above wrong among answered?
        c_auc = float("nan")
        sel = answered & ~np.isnan(conf)
        if sel.sum() > 10 and len(set(correct[sel])) == 2:
            c_auc = float(roc_auc_score(correct[sel], conf[sel]))
        maj = pd.Series(truth[has_truth]).value_counts(normalize=True).iloc[0]
        res.append({"field": name, "n_with_truth": n, "coverage": float(answered.sum() / n),
                    "accuracy_answered": acc, "majority_class_rate": float(maj), "confidence_auroc": c_auc})

    surv = o["label"] == "Survived"
    s = o[surv]
    report("roof_material (vs DINS, survived)", s["roof_material"],
           [ROOF_MAP.get(insp[g].get("roofconstruction") or "") for g in s["global_id"]], s["roof_material__c"])
    deck_truth = []
    for g in s["global_id"]:
        de, pc = insp[g].get("deckporchelevated"), insp[g].get("patiocovercarport")
        if de is None or pc is None:
            deck_truth.append(None)
        else:
            deck_truth.append("no" if (de == "No Deck/Porch" and pc == "No Patio Cover/Carport") else "yes")
    report("attached_deck_or_patio_cover (vs DINS, survived)", s["attached_deck_or_patio_cover"], deck_truth,
           s["attached_deck_or_patio_cover__c"])
    report("nearest_other_building (vs DINS points)", o["nearest_other_building"],
           [nn_bin(geom["nn_dist_m"].get(g)) for g in o["global_id"]], o["nearest_other_building__c"])
    report("buildings_within_50m (vs DINS points)", o["buildings_within_50m"],
           [n50_bin(geom["n_50m"].get(g)) if g in geom.index else None for g in o["global_id"]],
           o["buildings_within_50m__c"])

    if (ext["variant"] == "rot90").any():
        m = ext[ext["variant"] == "orig"].merge(ext[ext["variant"] == "rot90"], on="global_id", suffixes=("", "_r"))
        for k in SCHEMA:
            both = m[[k, k + "_r"]].dropna()
            res.append({"field": f"{k} (rotation invariance)", "n_with_truth": int(len(both)),
                        "coverage": float(((both[k] != CD) & (both[k + "_r"] != CD)).mean()) if len(both) else np.nan,
                        "accuracy_answered": float((both[k] == both[k + "_r"]).mean()) if len(both) else np.nan,
                        "majority_class_rate": float(both[k].value_counts(normalize=True).iloc[0]) if len(both) else np.nan,
                        "confidence_auroc": float("nan")})
    return res


def decision_head(ext: pd.DataFrame, recs: dict) -> list:
    o = ext[ext["variant"] == "orig"].copy()
    o["split"] = [recs[g]["split"] for g in o["global_id"]]
    o["y"] = [int(recs[g]["labels"]["outcome"] == "Destroyed") for g in o["global_id"]]
    o["event"] = [recs[g]["event"] for g in o["global_id"]]
    o["pilot_set"] = [recs[g].get("pilot_set") for g in o["global_id"]]
    o["structure_type"] = [recs[g]["inputs"]["prefire"].get("structure_type") or "NA" for g in o["global_id"]]
    o["year_built"] = [recs[g]["inputs"]["prefire"].get("year_built") for g in o["global_id"]]
    v = [recs[g]["inputs"]["prefire"].get("assessed_improved_value") for g in o["global_id"]]
    o["log_value"] = np.log1p(pd.to_numeric(pd.Series(v), errors="coerce")).values
    vis = [k for k in SCHEMA if k != "image_quality"]
    for k in vis:
        o[k] = o[k].fillna("NA")
    tr = o[o["split"] == "train"]
    if len(tr) < 100:
        return []
    out = []
    for name, cat, num in [("prefire_meta", ["structure_type"], ["year_built", "log_value"]),
                           ("vlm_attributes", vis, []),
                           ("vlm_attributes+prefire_meta", vis + ["structure_type"], ["year_built", "log_value"])]:
        tfs = [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat)]
        if num:
            tfs.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                         ("sc", StandardScaler())]), num))
        clf = Pipeline([("pre", ColumnTransformer(tfs)), ("lr", LogisticRegression(max_iter=3000, C=0.5))])
        clf.fit(tr[cat + num], tr["y"])
        o["p_" + name] = clf.predict_proba(o[cat + num])[:, 1]
        for ps in ("diagnostic", "natural"):
            te = o[(o["split"] == "test") & (o["pilot_set"] == ps)]
            if len(te) < 20 or len(set(te["y"])) < 2:
                continue
            y, p, ev = te["y"].values, te["p_" + name].values, te["event"].values
            ci = _cluster_boot(ev, lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan)
            out.append({"features": name, "eval": f"pilot {ps}", "n": int(len(te)),
                        "auroc": float(roc_auc_score(y, p)), "ci95": ci})
        pfiles = sorted(SPL.glob("pairs_test*.jsonl"))
        if pfiles:
            pm = o.set_index("global_id")["p_" + name]
            accs, evs = [], []
            for line in (ln for f in pfiles for ln in f.read_text().splitlines()):
                q = json.loads(line)
                gd, gs = q["destroyed"]["global_id"], q["survived"]["global_id"]
                if gd in pm.index and gs in pm.index:
                    accs.append(float(pm[gd] > pm[gs]) + 0.5 * float(pm[gd] == pm[gs]))
                    evs.append(q["event"])
            if accs:
                a, e = np.array(accs), np.array(evs)
                out.append({"features": name, "eval": "pairs (pair accuracy)", "n": len(a), "auroc": float(a.mean()),
                            "ci95": _cluster_boot(e, lambda i: a[i].mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="one or more run dirs (merged); first names the output")
    args = ap.parse_args()
    recs = load_records()
    ext = load_run(args.run)
    ext = ext[ext["global_id"].isin(recs)]
    ver = verification(ext, recs)
    head = decision_head(ext, recs)
    f = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
    L = [f"# Extraction analysis: `{' + '.join(args.run)}`\n",
         f"Parsed extractions: {len(ext)} ({(ext['variant'] == 'orig').sum()} orig).\n",
         "## Verifiability (independent references, never the fire outcome)\n",
         "| Field | N with reference | Coverage (answered) | Accuracy on answered | Majority-class rate | Confidence AUROC |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in ver:
        L.append(f"| {r['field']} | {r['n_with_truth']} | {f(r['coverage'])} | {f(r['accuracy_answered'])} | "
                 f"{f(r['majority_class_rate'])} | {f(r['confidence_auroc'])} |")
    if head:
        L += ["\n## Decision head (logistic regression fitted on train-event sample)\n",
              "| Features | Evaluated on | N | AUROC / pair acc [95% event CI] |", "|---|---|---:|---|"]
        for r in head:
            L.append(f"| {r['features']} | {r['eval']} | {r['n']} | {r['auroc']:.3f} [{f(r['ci95'][0])}, {f(r['ci95'][1])}] |")
    md = "\n".join(L) + "\n"
    (EXT / args.run[0] / "analysis.md").write_text(md)
    (EXT / args.run[0] / "analysis.json").write_text(json.dumps({"verification": ver, "head": head}, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    main()
