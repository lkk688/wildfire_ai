"""
Pre-registered confirmatory analysis (docs/CONFIRMATORY_PLAN.md).  Written and
committed BEFORE any model was run on the val-event confirmatory sets.

Reads the raw logs of the frozen runs and evaluates H1-H7 exactly as specified:
  primary CIs: pair/building-level bootstrap (val has 9 events, Glass = 61% of pairs,
  so event-cluster bootstrap is a sensitivity analysis); one-sided p-values from the
  bootstrap distribution; Holm correction over the primary family H2, H3, H4a.
  Ties (p == 0.5) in pairs get half credit (primary); strict scoring = sensitivity.

Output: output/experiments/confirmatory/results.{json,md}
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXP = ROOT / "output" / "experiments"
SPL = ROOT / "output" / "splits"
OUT = EXP / "confirmatory"
N_BOOT = 2000
SEED = 20261020
CORE = ["roofconstruction", "eaves", "ventscreen", "exteriorsiding", "windowpane",
        "deckporchongrade", "deckporchelevated", "fenceattachedtostructure"]


def best_rows(path: Path, key) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        k = key(r)
        if k not in out or r.get("parse_ok"):
            out[k] = r
    return out


def boot(n, fn, groups=None):
    rng = np.random.default_rng(SEED)
    vals = []
    if groups is None:
        for _ in range(N_BOOT):
            v = fn(rng.integers(0, n, n))
            if v is not None and not np.isnan(v):
                vals.append(v)
    else:
        u = np.unique(groups)
        idx = {g: np.where(groups == g)[0] for g in u}
        for _ in range(N_BOOT):
            v = fn(np.concatenate([idx[g] for g in rng.choice(u, len(u))]))
            if v is not None and not np.isnan(v):
                vals.append(v)
    return np.array(vals)


def summarize(est, vals, null=0.0):
    return {"estimate": float(est), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "p_one_sided": float((np.sum(vals <= null) + 1) / (len(vals) + 1))}


def pair_scores(run: str, pair_ids: set, strict=False):
    b = best_rows(EXP / "vlm_pairs" / run / "raw.jsonl", lambda r: (r["pair_id"], r["evidence"]))
    out = {}
    for (pid, ev), r in b.items():
        if pid not in pair_ids:
            continue
        if not r.get("parse_ok"):
            out[(pid, ev)] = np.nan
            continue
        pc = r["p_A_destroyed"] if r["destroyed_is"] == "A" else 1 - r["p_A_destroyed"]
        out[(pid, ev)] = float(pc > 0.5) + (0.0 if strict else 0.5 * float(pc == 0.5))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(line) for line in (SPL / "confirm_pairs.jsonl").read_text().splitlines()]
    pids = [p["pair_id"] for p in pairs]
    pev = np.array([p["event"] for p in pairs])
    direct = {json.loads(line)["global_id"]: json.loads(line)
              for line in (SPL / "confirm_direct.jsonl").read_text().splitlines()}
    res = {}

    # ---- pairs: H1 (estimation), H2 ----
    for run in ["confirm_pairs_sonnet", "confirm_pairs_qwen38"]:
        if not (EXP / "vlm_pairs" / run / "raw.jsonl").exists():
            continue
        for strict in (False, True):
            s = pair_scores(run, set(pids), strict)
            img = np.array([s.get((p, "image"), np.nan) for p in pids])
            meta = np.array([s.get((p, "meta"), np.nan) for p in pids])
            tag = f"{run}{'_strict' if strict else ''}"
            ok = ~np.isnan(img)
            res[f"H2 {tag}: image pair acc > 0.5"] = {
                **summarize(np.nanmean(img), boot(ok.sum(), lambda i: img[ok][i].mean()), 0.5),
                "n": int(ok.sum()), "missing": int((~ok).sum()),
                "event_cluster_ci95": list(np.percentile(boot(ok.sum(), lambda i: img[ok][i].mean(), pev[ok]), [2.5, 97.5])),
                "glass_excluded": float(np.nanmean(img[ok & ~np.char.startswith(pev.astype(str), "Glass")]))}
            both = ok & ~np.isnan(meta)
            d = img[both] - meta[both]
            res[f"H1 {tag}: image - meta pair acc (estimation)"] = {
                **summarize(d.mean(), boot(both.sum(), lambda i: d[i].mean())), "n": int(both.sum()),
                "meta_acc": float(meta[both].mean()),
                "event_cluster_ci95": list(np.percentile(boot(both.sum(), lambda i: d[i].mean(), pev[both]), [2.5, 97.5]))}

    # ---- H6: tabular reference (train-only) on the same pairs; H7: embedding head ----
    ref = EXP / "confirmatory" / "pair_references.json"
    if ref.exists():
        res["H6/H7 references (train-only fits)"] = json.loads(ref.read_text())

    # ---- direct: H3 (Sonnet image - mismatch), H5 (Qwen), H4 (Haiku inspector mechanism) ----
    def direct_auc(run, evidence):
        p = EXP / "vlm_conditions" / run / "raw.jsonl"
        if not p.exists():
            return None
        b = best_rows(p, lambda r: (r["global_id"], r["evidence"], r["context"]))
        return {g: r for (g, e, c), r in b.items() if e == evidence and c == "neutral" and g in direct}

    for run, h in [("confirm_direct_sonnet", "H3"), ("confirm_direct_qwen38", "H5")]:
        a, b = direct_auc(run, "image"), direct_auc(run, "image_mismatch")
        if not a or not b:
            continue
        ids = sorted(g for g in a if a[g].get("parse_ok") and g in b and b[g].get("parse_ok"))
        y = np.array([direct[g]["labels"]["outcome"] == "Destroyed" for g in ids], dtype=int)
        pa = np.array([a[g]["p_destroyed"] for g in ids])
        pb = np.array([b[g]["p_destroyed"] for g in ids])
        ev = np.array([direct[g]["event"] for g in ids])

        def f(i):
            if len(set(y[i])) < 2:
                return np.nan
            return roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i])
        res[f"{h} {run}: AUROC(image) - AUROC(mismatch)"] = {
            **summarize(f(np.arange(len(ids))), boot(len(ids), f)), "n": len(ids),
            "auroc_image": float(roc_auc_score(y, pa)), "auroc_mismatch": float(roc_auc_score(y, pb)),
            "missing": int(len(direct) - len(ids)),
            "event_cluster_ci95": list(np.percentile(boot(len(ids), f, ev), [2.5, 97.5]))}

    h = direct_auc("confirm_direct_haiku", "inspector")
    if h:
        ids = sorted(g for g in h if h[g].get("parse_ok"))
        y = np.array([direct[g]["labels"]["outcome"] == "Destroyed" for g in ids], dtype=int)
        p = np.array([h[g]["p_destroyed"] for g in ids])
        clean = np.array([all(direct[g]["inputs"]["inspector_postfire"].get(k) for k in CORE) for g in ids])
        nunk = np.array([sum(not direct[g]["inputs"]["inspector_postfire"].get(k) for k in CORE) for g in ids])
        full = boot(len(ids), lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan)
        res["H4a Haiku inspector: AUROC < 0.5 (full set)"] = {
            "estimate": float(roc_auc_score(y, p)), "ci95": list(np.percentile(full, [2.5, 97.5])),
            "p_one_sided": float((np.sum(full >= 0.5) + 1) / (len(full) + 1)), "n": len(ids)}
        if len(set(y[clean])) == 2:
            yc, pc_ = y[clean], p[clean]
            res["H4b Haiku inspector: AUROC on clean subset (descriptive)"] = {
                "estimate": float(roc_auc_score(yc, pc_)), "n": int(clean.sum()), "n_destroyed": int(yc.sum())}
        res["H4c corr(#Unknown, outcome) vs corr(#Unknown, Haiku p)"] = {
            "corr_unknown_outcome": float(np.corrcoef(nunk, y)[0, 1]),
            "corr_unknown_pred": float(np.corrcoef(nunk, p)[0, 1])}

    # Holm over primary family
    fam = [k for k in res if k.startswith(("H2 confirm_pairs_sonnet:", "H3 ", "H4a "))]
    ps = sorted((res[k]["p_one_sided"], k) for k in fam)
    for rank, (pv, k) in enumerate(ps):
        res[k]["p_holm"] = float(min(1.0, max(pv * (len(ps) - rank),
                                             *(res[kk].get("p_holm", 0) for _, kk in ps[:rank]))))
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    L = ["# Confirmatory results (val events, pre-registered)\n",
         "| Hypothesis | Estimate | 95% CI | p (one-sided) | p Holm | n / notes |", "|---|---:|---|---:|---:|---|"]
    for k, v in res.items():
        if "estimate" not in v:
            L.append(f"| {k} | | | | | {json.dumps(v)[:200]} |")
            continue
        ci = v.get("ci95", [np.nan, np.nan])
        extra = {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                 if kk not in ("estimate", "ci95", "p_one_sided", "p_holm")}
        L.append(f"| {k} | {v['estimate']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {v.get('p_one_sided', float('nan')):.4f} | "
                 f"{v.get('p_holm', float('nan')):.4f} | {json.dumps(extra)[:220]} |")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
