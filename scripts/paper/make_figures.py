"""
Paper figures, generated from the artefacts in output/ (no hand-copied numbers).

fig_see_vs_foresee.pdf  per model: pre-fire single-building AUROC (T1), pre-fire neighbour-pair
                        accuracy (T2), post-fire recognition AUROC (T3); 0.5 = chance on all three.
Run:  python3 scripts/paper/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper.make_tables import EXP, direct, md_table_value, pair_acc  # noqa: E402

OUT = ROOT / "paper" / "figs"
# validated categorical slots 1-3 (light surface); shapes are the secondary encoding
TASKS = [("T1 pre-fire, single building (AUROC)", "#2a78d6", "o"),
         ("T2 pre-fire, same-fire neighbour pairs (acc.)", "#eb6834", "s"),
         ("T3 post-fire recognition (AUROC)", "#1baf7a", "^")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def collect():
    comp = json.loads((EXP / "postfire" / "comparison.json").read_text())
    pf = {(r["set"], r["model"]): r["auroc"] for r in comp["rows"]}
    tp = ["pairs_test.jsonl", "pairs_test_large.jsonl"]
    emb = json.loads((EXP / "embedding_baseline" / "summary_facebook__dinov2-base.json").read_text())
    tab = json.loads((EXP / "tabular_baselines_pilot.json").read_text())
    tmeta = next(r for r in tab["runs"] if r["feature_set"] == "prefire_meta" and r["model"] == "hgb"
                 and r["pilot_set"] == "diagnostic")
    rows = [
        ("Claude Sonnet 5.5", direct("pilot_sonnet_diag")[0]["auroc"], pair_acc("pairs_sonnet", tp, "image")[0],
         pf[("subset400", "Claude Sonnet 5.5 zero-shot | post")]),
        ("Claude Haiku 5.5", direct("pilot_haiku_diag")[0]["auroc"], pair_acc("pairs_haiku", tp, "image")[0],
         pf[("subset400", "Claude Haiku 5.5 zero-shot | post")]),
        ("MiniMax-M3", direct("pilot_minimax_m3_diag")[0]["auroc"], pair_acc("pairs_minimax_m3", tp, "image")[0], None),
        ("Qwen3.8-27B (open)", direct("pilot_qwen38_diag")[0]["auroc"], pair_acc("pairs_qwen38", tp, "image")[0],
         pf[("full", "Qwen3.8-27B zero-shot | post")]),
        ("Qwen3.5-4B + LoRA (ours)", None, None, pf[("full", "Qwen3.5-4B + LoRA SFT (ours)")]),
        ("Frozen DINOv2 + LR", next(r for r in emb if r["eval"] == "pilot diagnostic")["score"],
         next(r for r in emb if r["eval"].startswith("pairs"))["score"],
         pf[("full", "lp_post facebook/dinov2-base (post-hoc stricter ref)")]),
        ("Tabular: year built, value", tmeta["auroc"], md_table_value(EXP / "pairs_reference.md", "prefire_meta |", 1), None),
    ]
    ceiling = md_table_value(EXP / "missingness_ablation.md", "values_random_imputed", 3)
    return rows, ceiling


def main() -> None:
    rows, ceiling = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.axvspan(0.5, ceiling, ymin=0, ymax=1, color="#f3eee6", zorder=0)
    ax.text(ceiling + 0.006, y.max() + 0.8, f"T2 oracle: inspector attributes ({ceiling:.2f})", ha="left", va="center",
            fontsize=6.5, color=MUTED)
    ax.text(0.494, -0.45, "chance", ha="right", va="bottom", fontsize=6.5, color=MUTED, rotation=90)
    for (label, color, marker), k in zip(TASKS, (1, 2, 3)):
        xs = [(yy, r[k]) for yy, r in zip(y, rows) if r[k] is not None]
        ax.scatter([v for _, v in xs], [yy for yy, _ in xs], s=34, marker=marker, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3, label=label)
    for yy, r in zip(y, rows):
        vals = [v for v in r[1:] if v is not None]
        ax.plot([min(vals), max(vals)], [yy, yy], color=GRID, lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.7, y.max() + 1.3)
    ax.set_xlabel("score (0.5 = chance)", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), frameon=False, fontsize=6.5, handletextpad=0.3,
              borderaxespad=0, ncol=1)
    fig.savefig(OUT / "fig_see_vs_foresee.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_see_vs_foresee.png", dpi=200, bbox_inches="tight")
    print("[OK]", OUT / "fig_see_vs_foresee.pdf")


if __name__ == "__main__":
    main()
