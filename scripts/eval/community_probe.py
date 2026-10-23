"""
(Pilot A) Frozen DINOv2-base linear probe on the community crops (zoom 16, raw image without the box), fitted on
2019-23 cells (train + val) for high_loss and applied to 2024-25 test cells. Run in the GPU env.
Output: output/community/dinov2_probe_test.json  {cell_id: p_high_loss}
"""
import json
import sys
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.embedding_baseline import embed  # noqa: E402

OUT = ROOT / "output" / "community"


def main():
    cells = [json.loads(x) for x in (OUT / "cells.jsonl").read_text().splitlines()]
    cells = [c for c in cells if c["imagery"]["accepted"]]
    tr = [c for c in cells if c["split"] in ("train", "val")]
    te = [c for c in cells if c["split"] == "test"]
    E = embed([c["imagery"]["image_path"] for c in tr + te], "facebook/dinov2-base", "cuda")
    clf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(64, len(tr) - 1), random_state=0)),
                    ("lr", LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=3000, scoring="roc_auc"))])
    clf.fit(E[: len(tr)], [c["high_loss"] for c in tr])
    p = clf.predict_proba(E[len(tr):])[:, 1]
    (OUT / "dinov2_probe_test.json").write_text(json.dumps({c["cell_id"]: float(v) for c, v in zip(te, p)}))
    print("train", len(tr), "test", len(te))


if __name__ == "__main__":
    main()
