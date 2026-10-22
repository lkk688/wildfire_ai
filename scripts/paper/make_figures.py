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
    plp = EXP / "prefire_sft_Qwen3.5-4B" / "log.json"
    prelora = json.loads(plp.read_text()).get("final", {}) if plp.exists() else {}
    rows = [
        ("Claude Sonnet 5.5", direct("pilot_sonnet_diag")[0]["auroc"], pair_acc("pairs_sonnet", tp, "image")[0],
         pf[("subset400", "Claude Sonnet 5.5 zero-shot | post")]),
        ("Claude Haiku 5.5", direct("pilot_haiku_diag")[0]["auroc"], pair_acc("pairs_haiku", tp, "image")[0],
         pf[("subset400", "Claude Haiku 5.5 zero-shot | post")]),
        ("MiniMax-M3", direct("pilot_minimax_m3_diag")[0]["auroc"], pair_acc("pairs_minimax_m3", tp, "image")[0], None),
        ("Qwen3.8-27B (open)", direct("pilot_qwen38_diag")[0]["auroc"], pair_acc("pairs_qwen38", tp, "image")[0],
         pf[("subset400", "Qwen3.8-27B zero-shot | post")]),
        ("Qwen3.5-4B + LoRA (ours)", prelora.get("T1_auroc"), prelora.get("T2_test_pair_acc"),
         pf[("subset400", "Qwen3.5-4B + LoRA SFT (ours)")]),
        ("Frozen DINOv2 + LR", next(r for r in emb if r["eval"] == "pilot diagnostic")["score"],
         next(r for r in emb if r["eval"].startswith("pairs"))["score"],
         pf[("subset400", "lp_post facebook/dinov2-base (post-hoc stricter ref)")]),
        ("Tabular: type, year, value", tmeta["auroc"], md_table_value(EXP / "pairs_reference.md", "prefire_meta |", 1), None),
    ]
    ceiling = json.loads((EXP / "oracle_same_pairs.json").read_text())["values, random-imputed"]["all"]
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
    ax.text(ceiling + 0.006, y.max() + 0.8, f"T2 oracle: attribute values (≥{ceiling:.2f})", ha="left", va="center",
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


# ---------------------------------------------------------------- mechanism figure
def vlm_auc(run, evidence):
    from sklearn.metrics import roc_auc_score
    p = EXP / "vlm_conditions" / run / "raw.jsonl"
    best = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] == evidence and (r["global_id"] not in best or r.get("parse_ok")):
            best[r["global_id"]] = r
    ok = [r for r in best.values() if r.get("parse_ok")]
    if len(ok) < 100:
        return None
    return roc_auc_score([r["label"] == "Destroyed" for r in ok], [r["p_destroyed"] for r in ok])


def mechanism():
    ca = json.loads((EXP / "context_ablation.json").read_text())
    order = [("center_19m", "roof\n(19 m)"), ("center_64m", "centre\n(64 m)"), ("full", "full\n(150 m)"),
             ("masked_19m", "building\nmasked"), ("masked_37m", "bldg+yard\nmasked"), ("ring_only", "outer\nring")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(order))
    for (label, color, marker), key, cik in zip(TASKS, ("T1_auroc", "T2_pair_acc", "T3_auroc"), ("T1_ci", "T2_ci", "T3_ci")):
        v = [ca[k][key] for k, _ in order]
        lo = [ca[k][cik][0] for k, _ in order]
        hi = [ca[k][cik][1] for k, _ in order]
        a1.plot(x, v, color=color, lw=2, zorder=2)
        a1.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
        a1.scatter(x, v, color=color, marker=marker, s=26, edgecolor="white", linewidth=1, zorder=3, label=label.split(" (")[0])
    for run, ev, xi, color, marker in [("pilot_sonnet_diag", "image", 2, TASKS[0][1], "o"),
                                       ("pilot_sonnet_diag", "image_masked", 4, TASKS[0][1], "o"),
                                       ("postfire_sonnet", "postfire_image", 2, TASKS[2][1], "^"),
                                       ("postfire_sonnet", "postfire_masked", 4, TASKS[2][1], "^"),
                                       ("pilot_qwen38_diag", "image_masked", 4, TASKS[0][1], "D"),
                                       ("postfire_qwen38", "postfire_masked", 4, TASKS[2][1], "D")]:
        v = vlm_auc(run, ev)
        if v is not None:
            a1.scatter([xi + 0.12], [v], facecolor="white", edgecolor=color, marker=marker, s=26, linewidth=1.4, zorder=4)
    a1.scatter([], [], facecolor="white", edgecolor=MUTED, marker="o", s=26, linewidth=1.4, label="Claude Sonnet (open)")
    a1.scatter([], [], facecolor="white", edgecolor=MUTED, marker="D", s=22, linewidth=1.4, label="Qwen3.8-27B (open)")
    a1.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    a1.set_xticks(x)
    a1.set_xticklabels([l for _, l in order], fontsize=6.5)
    a1.set_ylim(0.45, 1.0)
    a1.set_ylabel("score (0.5 = chance)")
    a1.set_title("(a) Where the signal lives (frozen DINOv2 + LR)", fontsize=8, loc="left")
    a1.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, handletextpad=0.2,
              columnspacing=0.8)

    sizes = [100, 400, 800, 1646]
    sft = []
    sft_sd = []
    for n in sizes:
        if n == 1646:
            v = [json.loads((EXP / "postfire" / "sft_Qwen3.5-4B" / "log.json").read_text())["test"]["auroc"]]
        else:
            v = [json.loads(f.read_text())["test"]["auroc"] for f in (EXP / "postfire").glob(f"sft_Qwen3.5-4B_n{n}_s*/log.json")]
        sft.append(float(np.mean(v)))
        sft_sd.append(float(np.std(v)))
    lp = json.loads((EXP / "postfire" / "lp_efficiency.json").read_text())
    lpm = [np.mean(lp[str(n)]) for n in sizes]
    lps = [np.std(lp[str(n)]) for n in sizes]
    comp = json.loads((EXP / "postfire" / "comparison.json").read_text())
    base = next(r["auroc"] for r in comp["rows"] if r["set"] == "full" and r["model"] == "Qwen3.5-4B base (same readout)")
    a2.errorbar(sizes, sft, yerr=sft_sd, color=TASKS[2][1], lw=2, marker="^", ms=5, mec="white", capsize=2,
                label="Qwen3.5-4B + LoRA SFT (3 seeds)")
    a2.errorbar(sizes, lpm, yerr=lps, color="#2a78d6", lw=2, marker="o", ms=4, mec="white", capsize=2,
                label="frozen DINOv2 + LR (5 seeds)")
    a2.axhline(base, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    a2.text(1646, base + 0.004, "4B zero-shot (same readout)", fontsize=6, color=MUTED, ha="right")
    a2.set_xscale("log")
    a2.set_xticks(sizes)
    a2.set_xticklabels([str(s) for s in sizes])
    a2.set_xlabel("training buildings (2019–23 fires)")
    a2.set_ylabel("post-fire test AUROC")
    a2.set_title("(b) Data efficiency, post-fire", fontsize=8, loc="left")
    a2.legend(fontsize=6, frameon=False, loc="upper left")
    for ax in (a1, a2):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_mechanism.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_mechanism.png", dpi=200, bbox_inches="tight")
    print("[OK]", OUT / "fig_mechanism.pdf")


if __name__ == "__main__":
    mechanism()


# ---------------------------------------------------------------- overview figure
def overview():
    import datetime as dt
    from PIL import Image as PImage
    sp = ROOT / "output" / "splits"
    pf = {json.loads(l)["global_id"]: json.loads(l) for l in (sp / "postfire_test.jsonl").read_text().splitlines()}
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in (sp / "pairs_test.jsonl").read_text().splitlines()}
    b = pf["6bfd53ac-2552-4df1-b13e-83de88714470"]
    q = pairs["p0041"]
    fig = plt.figure(figsize=(6.9, 1.75))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.55, 1, 1, 0.15, 1, 1], wspace=0.12)

    def thumb(ax, path, title, sub=None, color=None):
        im = PImage.open(path).convert("RGB").crop((96, 96, 416, 416))
        ax.imshow(im)
        ax.add_patch(plt.Circle((160, 160), 26, fill=False, ec=color or "white", lw=1.4))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=6.3, pad=2)
        if sub:
            ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6, color=MUTED)

    ax = fig.add_subplot(gs[0, 0])
    fire = dt.date.fromisoformat(b["imagery"]["fire_start"])
    ev = [("capture\n(pre)", dt.date.fromisoformat(b["imagery"]["capture_date"]), "#2a78d6"),
          ("Wayback\nrelease", dt.date.fromisoformat(b["imagery"]["release_date"]), MUTED),
          ("fire", fire, "#eb6834"),
          ("capture\n(post)", dt.date.fromisoformat(b["imagery_postfire"]["capture_date"]), "#1baf7a")]
    t0 = min(d for _, d, _ in ev)
    xs = [(d - t0).days for _, d, _ in ev]
    ax.plot([min(xs) - 20, max(xs) + 20], [0, 0], color=INK, lw=1)
    for (lab, d, c), x, yy in zip(ev, xs, (0.35, -0.35, 0.35, -0.35)):
        ax.scatter([x], [0], color=c, s=28, zorder=3, edgecolor="white", linewidth=1)
        ax.text(x, yy, f"{lab}\n{d:%Y-%m-%d}", ha="center", va="center", fontsize=5.8, color=INK)
    span = max(xs) - min(xs)
    ax.set_ylim(-0.75, 0.75); ax.set_xlim(min(xs) - 0.12 * span, max(xs) + 0.12 * span); ax.axis("off")
    ax.set_title("(a) per-location provenance", fontsize=7, pad=2)
    thumb(fig.add_subplot(gs[0, 1]), q["destroyed"]["imagery"]["image_path"], "(b) T2: destroyed", "pre-fire", "#eb6834")
    thumb(fig.add_subplot(gs[0, 2]), q["survived"]["imagery"]["image_path"], f"its neighbour ({q['dist_m']:.0f} m)",
          "pre-fire", "white")
    thumb(fig.add_subplot(gs[0, 4]), b["imagery"]["image_path"], "(c) T1: pre-fire", "time-verified")
    thumb(fig.add_subplot(gs[0, 5]), b["imagery_postfire"]["image_path"], "T3: post-fire", "first capture after fire")
    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fig_overview.png", dpi=200, bbox_inches="tight")
    print("[OK]", OUT / "fig_overview.pdf")


if __name__ == "__main__":
    overview()
