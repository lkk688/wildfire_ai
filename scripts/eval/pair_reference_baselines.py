"""
Tabular reference scores on the exposure-controlled pair sets
(output/splits/pairs_test.jsonl + pairs_test_large.jsonl; the XL set is evaluated in c1_scaleup.py): HGB trained on 2019-2023 train/val events.

Output: output/experiments/pairs_reference.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.tabular_baselines_on_pilot import GEOM, GEOM_COLS, INSPECTOR, SPLITS, SRC, model, prep  # noqa: E402
from scripts.eval.vlm_metrics import _cluster_boot  # noqa: E402

CORE = ["ROOFCONSTRUCTION", "EAVES", "VENTSCREEN", "EXTERIORSIDING", "WINDOWPANE",
        "DECKPORCHONGRADE", "DECKPORCHELEVATED", "FENCEATTACHEDTOSTRUCTURE"]


def main() -> None:
    df = prep(pd.read_parquet(SRC).merge(pd.read_parquet(SPLITS)[["GLOBALID", "split", "y", "event"]], on="GLOBALID"))
    df = df.merge(pd.read_parquet(GEOM), on="GLOBALID", how="left")
    df["log_nn_dist"] = np.log1p(df["nn_dist_m"])
    tr = df[df["split"].isin(["train", "val"])]
    D = df.set_index("GLOBALID")
    pfiles = [ROOT / "output" / "splits" / f for f in ("pairs_test.jsonl", "pairs_test_large.jsonl")]
    pairs = [json.loads(line) for f in pfiles for line in f.read_text().splitlines()]
    gd = [p["destroyed"]["global_id"] for p in pairs]
    gs = [p["survived"]["global_id"] for p in pairs]
    ev = np.array([p["event"] for p in pairs])
    both_obs = ((D.loc[gd, CORE] != "UNKNOWN").all(axis=1).values & (D.loc[gs, CORE] != "UNKNOWN").all(axis=1).values)
    num = ["YEARBUILT", "log_value"]
    L = ["# Tabular references on the pair sets\n",
         f"{len(pairs)} pairs from {', '.join(f.name for f in pfiles)}; HGB fitted on 2019-2023 events. "
         f"Clean subset = both homes have all {len(CORE)} core inspector attributes recorded "
         f"({both_obs.sum()} pairs).\n",
         "| Features | Pair acc, all [95% event CI] | Pair acc, clean subset |", "|---|---|---:|"]
    for name, cat, nm in [("prefire_meta", ["STRUCTURETYPE"], num),
                          ("prefire_meta+geometry", ["STRUCTURETYPE"], num + GEOM_COLS),
                          ("inspector (leaky oracle)", ["STRUCTURETYPE"] + INSPECTOR, num)]:
        m = model("hgb", cat, nm).fit(tr[cat + nm], tr["y"])
        a = m.predict_proba(D.loc[gd][cat + nm])[:, 1]
        b = m.predict_proba(D.loc[gs][cat + nm])[:, 1]
        acc = (a > b) + 0.5 * (a == b)
        ci = _cluster_boot(ev, lambda i: acc[i].mean())
        L.append(f"| {name} | {acc.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {acc[both_obs].mean():.3f} |")
    md = "\n".join(L) + "\n"
    (ROOT / "output" / "experiments" / "pairs_reference.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
