"""
Metrics for the controlled VLM condition runs (run_vlm_conditions.py).

Naming is explicit because the legacy code mixed two denominators:
  FPR               = FP / (FP + TN)   share of *survived* homes predicted destroyed
  false_alarm_ratio = FP / (TP + FP)   share of *destroyed predictions* that were wrong
Uncertainty: buildings in one fire are not independent, so 95% intervals come
from a cluster bootstrap that resamples whole fire events.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
SEED = 1019


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "balanced_accuracy": (tpr + tnr) / 2,
        "TPR_recall_destroyed": tpr,
        "FPR": fp / (fp + tn) if fp + tn else float("nan"),
        "false_alarm_ratio": fp / (tp + fp) if tp + fp else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 and len(set(p)) > 1 else
                 (0.5 if len(set(y)) == 2 else float("nan")),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "mean_p_destroyed_given_survived": float(p[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mean_p_destroyed_given_destroyed": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
    }


def _cluster_boot(events: np.ndarray, fn, n_boot: int = N_BOOT) -> tuple:
    rng = np.random.default_rng(SEED)
    uniq = np.unique(events)
    idx_by = {e: np.where(events == e)[0] for e in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[e] for e in pick])
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def load_rows(raw_path: Path) -> dict:
    """(global_id, evidence, context) -> latest row, preferring parsed rows."""
    best = {}
    for line in Path(raw_path).read_text().splitlines():
        r = json.loads(line)
        k = (r["global_id"], r["evidence"], r["context"])
        if k not in best or r.get("parse_ok") or not best[k].get("parse_ok"):
            best[k] = r
    return best


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def summarize_run(raw_path: Path) -> dict:
    rows = load_rows(raw_path)
    conds = defaultdict(list)
    for (gid, e, c), r in rows.items():
        conds[(e, c)].append(r)
    models = sorted({str(r.get("reported_model")) for r in rows.values()})

    per_cond = {}
    for (e, c), rs in sorted(conds.items()):
        ok = [r for r in rs if r.get("parse_ok")]
        y = np.array([r["label"] == "Destroyed" for r in ok], dtype=int)
        p = np.array([r["p_destroyed"] for r in ok], dtype=float)
        ev = np.array([r["event"] for r in ok])
        m = binary_metrics(y, p) if len(ok) else {}
        if len(ok) and len(set(y)) == 2:
            m["auroc_ci95"] = _cluster_boot(ev, lambda i: binary_metrics(y[i], p[i])["auroc"])
            m["balanced_accuracy_ci95"] = _cluster_boot(ev, lambda i: binary_metrics(y[i], p[i])["balanced_accuracy"])
            m["FPR_ci95"] = _cluster_boot(ev, lambda i: binary_metrics(y[i], p[i])["FPR"])
        per_ev = {}
        for e_name in sorted(set(ev)):
            sel = ev == e_name
            if len(set(y[sel])) == 2:
                per_ev[e_name] = {"n": int(sel.sum()), "auroc": binary_metrics(y[sel], p[sel])["auroc"],
                                  "FPR": binary_metrics(y[sel], p[sel])["FPR"]}
        m.update({"n_requested": len(rs), "n_parsed": len(ok),
                  "parse_fail_rate": 1 - len(ok) / len(rs) if rs else float("nan"), "per_event": per_ev})
        per_cond[f"{e}|{c}"] = m

    # Paired contrasts on buildings parsed under both conditions.
    def paired(a, b, stat):
        ka = {r["global_id"]: r for r in conds.get(a, []) if r.get("parse_ok")}
        kb = {r["global_id"]: r for r in conds.get(b, []) if r.get("parse_ok")}
        ids = sorted(set(ka) & set(kb))
        if len(ids) < 10:
            return None
        y = np.array([ka[i]["label"] == "Destroyed" for i in ids], dtype=int)
        pa = np.array([ka[i]["p_destroyed"] for i in ids])
        pb = np.array([kb[i]["p_destroyed"] for i in ids])
        ev = np.array([ka[i]["event"] for i in ids])

        def f(idx):
            return stat(y[idx], pa[idx]) - stat(y[idx], pb[idx])
        return {"a": "|".join(a), "b": "|".join(b), "n": len(ids),
                "delta": float(f(np.arange(len(ids)))), "ci95": _cluster_boot(ev, f)}

    auroc = lambda y, p: binary_metrics(y, p)["auroc"]  # noqa: E731
    fpr = lambda y, p: binary_metrics(y, p)["FPR"]  # noqa: E731
    mean_p_surv = lambda y, p: float(p[y == 0].mean()) if (y == 0).any() else float("nan")  # noqa: E731
    contrasts = []
    for c in ("neutral", "extreme"):
        for a, b in [("image", "image_mismatch"), ("image", "meta"), ("inspector", "meta")]:
            r = paired((a, c), (b, c), auroc)
            if r:
                contrasts.append({"question": f"visual/attr increment ΔAUROC ({a} − {b}, {c})", **r})
    for e in ("meta", "image", "image_mismatch", "inspector"):
        for name, st in [("ΔFPR", fpr), ("Δ mean p(destroyed | survived)", mean_p_surv), ("ΔAUROC", auroc)]:
            r = paired((e, "extreme"), (e, "neutral"), st)
            if r:
                contrasts.append({"question": f"prompt bias {name} (extreme − neutral, {e})", **r})

    L = [f"# VLM condition run: `{Path(raw_path).parent.name}`\n",
         f"Reported model id(s): {', '.join(models)}. Metrics on parsed responses; threshold p>=0.5. "
         "FPR = FP/(FP+TN); false-alarm ratio = FP/(TP+FP). 95% CI = fire-event cluster bootstrap.\n",
         "| Evidence | Context | Parsed/Req | Pos rate | AUROC [95% CI] | Bal. acc | FPR [95% CI] | "
         "False-alarm ratio | Brier | mean p(D) Surv / Destr |",
         "|---|---|---:|---:|---|---:|---|---:|---:|---|"]
    for k, m in per_cond.items():
        e, c = k.split("|")
        if not m.get("n"):
            L.append(f"| {e} | {c} | 0/{m['n_requested']} | | | | | | | |")
            continue
        ci = m.get("auroc_ci95", (np.nan, np.nan))
        fci = m.get("FPR_ci95", (np.nan, np.nan))
        L.append(f"| {e} | {c} | {m['n_parsed']}/{m['n_requested']} | {_fmt(m['pos_rate'], True)} | "
                 f"{_fmt(m['auroc'])} [{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(m['balanced_accuracy'])} | "
                 f"{_fmt(m['FPR'], True)} [{_fmt(fci[0], True)}, {_fmt(fci[1], True)}] | "
                 f"{_fmt(m['false_alarm_ratio'], True)} | {_fmt(m['brier'])} | "
                 f"{_fmt(m['mean_p_destroyed_given_survived'])} / {_fmt(m['mean_p_destroyed_given_destroyed'])} |")
    if contrasts:
        L += ["\n## Paired contrasts (same buildings)\n", "| Question | N | Δ | 95% CI |", "|---|---:|---:|---|"]
        for r in contrasts:
            L.append(f"| {r['question']} | {r['n']} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
    L += ["\n## Per-event AUROC\n"]
    evs = sorted({e for m in per_cond.values() for e in m.get("per_event", {})})
    if evs:
        L.append("| Condition | " + " | ".join(e.split(" | ")[0] for e in evs) + " |")
        L.append("|---|" + "---:|" * len(evs))
        for k, m in per_cond.items():
            L.append(f"| {k} | " + " | ".join(_fmt(m["per_event"].get(e, {}).get("auroc")) for e in evs) + " |")
    return {"reported_models": models, "conditions": per_cond, "contrasts": contrasts, "markdown": "\n".join(L) + "\n"}
