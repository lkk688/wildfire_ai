"""
Oracle variants on exactly the 403 test pairs the VLMs were scored on (review W1/W3):
HGB fitted on 2019-23 train/val events with inspector attributes as
  raw (Unknown as category) | missingness only | mode-imputed values | random-imputed values,
and a complete-case subset (both homes fully recorded).  Pair acc with ties = 0.5,
fire-event cluster bootstrap CI.  Output: output/experiments/oracle_same_pairs.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.tabular_baselines_on_pilot import INSPECTOR, SPLITS, SRC, model, prep  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

SPL = ROOT / "output" / "splits"


def main():
    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    trm = df["split"].isin(["train", "val"])
    rng = np.random.default_rng(1019)
    for a in INSPECTOR:
        obs = df.loc[trm & (df[a] != "UNKNOWN"), a]
        miss = df[a] == "UNKNOWN"
        df[a + "__imp"] = df[a].where(~miss, obs.mode().iloc[0])
        df[a + "__rnd"] = df[a]
        df.loc[miss, a + "__rnd"] = rng.choice(obs.values, size=int(miss.sum()))
        df[a + "__miss"] = miss.astype(int)
    df["n_missing"] = df[[a + "__miss" for a in INSPECTOR]].sum(axis=1)
    tr = df[trm]
    D = df.set_index("GLOBALID")
    pairs = [json.loads(l) for f in ("pairs_test.jsonl", "pairs_test_large.jsonl") for l in (SPL / f).read_text().splitlines()]
    gd = [p["destroyed"]["global_id"] for p in pairs]
    gs = [p["survived"]["global_id"] for p in pairs]
    ev = np.array([p["event"] for p in pairs])
    full = ((D.loc[gd, INSPECTOR] != "UNKNOWN").all(axis=1).values & (D.loc[gs, INSPECTOR] != "UNKNOWN").all(axis=1).values)
    num = ["YEARBUILT", "log_value"]
    sets = {"prefire metadata": (["STRUCTURETYPE"], num),
            "raw (Unknown as category)": (["STRUCTURETYPE"] + INSPECTOR, num),
            "missingness only": (["STRUCTURETYPE"], num + ["n_missing"] + [a + "__miss" for a in INSPECTOR]),
            "values, mode-imputed": (["STRUCTURETYPE"] + [a + "__imp" for a in INSPECTOR], num),
            "values, random-imputed": (["STRUCTURETYPE"] + [a + "__rnd" for a in INSPECTOR], num)}
    res = {"n_pairs": len(pairs), "n_complete_case": int(full.sum())}
    for name, (cat, nm) in sets.items():
        m = model("hgb", cat, nm).fit(tr[cat + nm], tr["y"])
        a_ = m.predict_proba(D.loc[gd][cat + nm])[:, 1]
        b_ = m.predict_proba(D.loc[gs][cat + nm])[:, 1]
        acc = (a_ > b_) + 0.5 * (a_ == b_)
        res[name] = {"all": float(acc.mean()), "ci95": _cluster_boot(ev, lambda i: acc[i].mean()),
                     "complete_case": float(acc[full].mean())}
        print(name, res[name])
    (ROOT / "output" / "experiments" / "oracle_same_pairs.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
