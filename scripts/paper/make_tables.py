"""
Generate the paper's main tables and in-text number macros directly from the
experiment artefacts in output/ (no hand-copied numbers).

Outputs:
  paper/tables/tab_prefire.tex    pre-fire: direct judgment, image increment, pairs (test + val confirmation)
  paper/tables/tab_postfire.tex   post-fire: zero-shot, supervised, SFT, retrieval
  paper/numbers.tex               \\newcommand macros for numbers quoted in the text
Run:  python3 scripts/paper/make_tables.py
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "output" / "experiments"
SPL = ROOT / "output" / "splits"
PAPER = ROOT / "paper"
SEED = 1019


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}".replace("0.", ".", 1) if abs(x) < 1 else f"{x:.3f}"


def ci(lo, hi):
    return f"{{\\scriptsize[{f3(lo)},{f3(hi)}]}}"


def sgn(x):
    return f"{x:+.3f}".replace("0.", ".", 1)


# ------------------------------------------------------------------ pre-fire
def direct(run):
    p = EXP / "vlm_conditions" / run / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    c = s["conditions"].get("image|neutral")
    inc = next((x for x in s["contrasts"] if x["a"] == "image|neutral" and x["b"] == "image_mismatch|neutral"), None)
    return c, inc


def pair_acc(run, pair_files, evidence):
    p = EXP / "vlm_pairs" / run / "raw.jsonl"
    if not p.exists():
        return None
    ids = {json.loads(l)["pair_id"]: json.loads(l)["event"] for f in pair_files for l in (SPL / f).read_text().splitlines()}
    best = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["evidence"] != evidence or r["pair_id"] not in ids:
            continue
        if r["pair_id"] not in best or r.get("parse_ok"):
            best[r["pair_id"]] = r
    ok = [r for r in best.values() if r.get("parse_ok")]
    if len(ok) < 0.9 * len(ids):
        return None
    pc = np.array([r["p_A_destroyed"] if r["destroyed_is"] == "A" else 1 - r["p_A_destroyed"] for r in ok])
    acc = (pc > 0.5) + 0.5 * (pc == 0.5)
    ev = np.array([r["event"] for r in ok])
    rng = np.random.default_rng(SEED)
    u = np.unique(ev)
    idx = {e: np.where(ev == e)[0] for e in u}
    b = [acc[np.concatenate([idx[e] for e in rng.choice(u, len(u))])].mean() for _ in range(1000)]
    return float(acc.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(ok)


def md_table_value(path, row_startswith, col):
    for line in Path(path).read_text().splitlines():
        if line.startswith("| " + row_startswith):
            cells = [c.strip() for c in line.strip("|").split("|")]
            return float(re.match(r"[-+]?[0-9.]+", cells[col]).group(0))
    return None


def prefire_table(macros):
    test_pairs = ["pairs_test.jsonl", "pairs_test_large.jsonl"]
    val_pairs = ["confirm_pairs.jsonl"]
    conf = json.loads((EXP / "confirmatory" / "results.json").read_text())
    rows = []
    for name, d_run, p_run, cd_run, cp_run in [
        ("Claude Sonnet 5.5", "pilot_sonnet_diag", "pairs_sonnet", "confirm_direct_sonnet", "confirm_pairs_sonnet"),
        ("Claude Haiku 5.5", "pilot_haiku_diag", "pairs_haiku", None, None),
        ("MiniMax-M3", "pilot_minimax_m3_diag", "pairs_minimax_m3", None, None),
        ("Qwen3.8-27B (open)", "pilot_qwen38_diag", "pairs_qwen38", "confirm_direct_qwen38", "confirm_pairs_qwen38")]:
        c, inc = direct(d_run)
        pi = pair_acc(p_run, test_pairs, "image")
        pm = pair_acc(p_run, test_pairs, "meta")
        cv = None
        if cd_run:
            key = next(k for k in conf if k.startswith(("H3 ", "H5 ")) and cd_run in k)
            cv = conf[key]
        cpv = pair_acc(cp_run, val_pairs, "image") if cp_run else None
        rows.append((name, c, inc, pi, pm, cv, cpv))
        tag = re.sub(r"[^A-Za-z]", "", name.split("(")[0])
        macros[f"pre{tag}Auc"] = f3(c["auroc"])
        macros[f"pre{tag}Inc"] = sgn(inc["delta"])
        if pi:
            macros[f"pair{tag}Img"] = f3(pi[0])
        if cpv:
            macros[f"pairVal{tag}Img"] = f3(cpv[0])
        if cv:
            macros[f"preVal{tag}Inc"] = sgn(cv["estimate"])

    tab = json.loads((EXP / "tabular_baselines_pilot.json").read_text())
    tmeta = next(r for r in tab["runs"] if r["feature_set"] == "prefire_meta" and r["model"] == "hgb" and r["pilot_set"] == "diagnostic")
    emb = json.loads((EXP / "embedding_baseline" / "summary_facebook__dinov2-base.json").read_text())
    emb_d = next(r for r in emb if r["eval"] == "pilot diagnostic")
    emb_p = next(r for r in emb if r["eval"].startswith("pairs"))
    ref = json.loads((EXP / "confirmatory" / "pair_references.json").read_text())
    pref_meta = md_table_value(EXP / "pairs_reference.md", "prefire_meta |", 1)
    miss_rand_pair = md_table_value(EXP / "missingness_ablation.md", "values_random_imputed", 3)
    miss_only_pair = md_table_value(EXP / "missingness_ablation.md", "missing_only", 3)
    raw_pair = md_table_value(EXP / "missingness_ablation.md", "raw (Unknown", 3)
    macros["ceilRawPair"] = f3(raw_pair)
    macros.update({"refMetaPair": f3(pref_meta), "refDinoAuc": f3(emb_d["score"]), "refDinoPair": f3(emb_p["score"]),
                   "ceilValuePair": f3(miss_rand_pair), "ceilMissPair": f3(miss_only_pair),
                   "refMetaAuc": f3(tmeta["auroc"])})

    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l cc cc cc}", r"\toprule",
         r" & \multicolumn{2}{c}{Single building (AUROC)} & \multicolumn{2}{c}{Neighbour pairs, test (acc.)} & \multicolumn{2}{c}{Untouched val events} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Model & image & $\Delta$ vs.\ mismatched & image+meta & meta only & $\Delta$ AUROC (direct) & pair acc. \\",
         r"\midrule"]
    for name, c, inc, pi, pm, cv, cpv in rows:
        L.append(" & ".join([
            name,
            f"{f3(c['auroc'])} {ci(*c['auroc_ci95'])}",
            f"{sgn(inc['delta'])} {ci(*inc['ci95'])}",
            f"{f3(pi[0])} {ci(pi[1], pi[2])}" if pi else "--",
            f"{f3(pm[0])}" if pm else "--",
            f"{sgn(cv['estimate'])} {ci(*cv['ci95'])}" if cv else "--",
            f"{f3(cpv[0])} {ci(cpv[1], cpv[2])}" if cpv else "--"]) + r" \\")
    L += [r"\midrule",
          " & ".join(["Tabular, pre-fire metadata (HGB)", f3(tmeta["auroc"]), "--", f3(pref_meta), "--", "--",
                      f3(ref["H6 tabular prefire_meta (train-only) pair acc"])]) + r" \\",
          " & ".join(["Frozen DINOv2 + LR", f3(emb_d["score"]), "--", f3(emb_p["score"]), "--", "--",
                      f3(ref["H7 DINOv2 head pair acc confirm_pairs"])]) + r" \\",
          " & ".join([r"Inspector attributes, values only$^\dagger$", "--", "--", f3(miss_rand_pair), "--", "--", "--"]) + r" \\",
          " & ".join([r"Inspector attributes, missingness only$^\dagger$", "--", "--", f3(miss_only_pair), "--", "--", "--"]) + r" \\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Pre-fire structure loss.} Zero-shot VLMs on time-verified pre-fire crops of held-out 2024--25 fires "
          r"(238 balanced buildings; 403 same-fire neighbour pairs 15--100\,m apart), and pre-registered replication on untouched "
          r"2019--23 validation fires (220 buildings; 289 pairs). $\Delta$ vs.\ mismatched: AUROC with the building's own image minus "
          r"with another building's image from the same fire. Brackets: 95\% fire-event cluster bootstrap. "
          r"$^\dagger$Post-fire inspector records (not available before a fire): attribute values with unknowns randomly imputed vs.\ "
          r"only the pattern of missing fields, all 2024--25 test pairs within 100\,m.}",
          r"\label{tab:prefire}", r"\end{table*}"]
    return "\n".join(L)


# ------------------------------------------------------------------ post-fire
def postfire_table(macros):
    comp = json.loads((EXP / "postfire" / "comparison.json").read_text())
    rows = {(r["set"], r["model"]): r for r in comp["rows"]}
    ret = EXP / "postfire" / "retrieval" / "report.json"
    retr = {}
    if ret.exists():
        for r in json.loads(ret.read_text())["rows"]:
            retr[(r["set"], r["model"])] = r
    ov = json.loads((EXP / "postfire" / "output_value.json").read_text())
    spec = [
        ("Zero-shot", "Claude Sonnet 5.5 (post / pre+post)", ("Claude Sonnet 5.5 zero-shot | post", "Claude Sonnet 5.5 zero-shot | pre+post")),
        ("Zero-shot", "Claude Haiku 5.5 (post / pre+post)", ("Claude Haiku 5.5 zero-shot | post", "Claude Haiku 5.5 zero-shot | pre+post")),
        ("Zero-shot", "Qwen3.8-27B (post / pre+post)", ("Qwen3.8-27B zero-shot | post", "Qwen3.8-27B zero-shot | pre+post")),
        ("Zero-shot", "Qwen3.5-4B, generated $p$", ("Qwen3.5-4B zero-shot (generative) | post",)),
        ("Zero-shot", "Qwen3.5-4B, token-prob.\\ readout", ("Qwen3.5-4B base (same readout)",)),
        ("Supervised", "DINOv2-small fine-tuned", ("ft DINOv2-small (pre-registered S2 ref)",)),
        ("Supervised", "Frozen DINOv2 + LR (post / pre+post)", ("lp_post facebook/dinov2-base (post-hoc stricter ref)", "lp_prepost facebook/dinov2-base (post-hoc stricter ref)")),
        ("SFT (ours)", "Qwen3.5-4B + LoRA", ("Qwen3.5-4B + LoRA SFT (ours)",)),
        ("SFT (ours)", "Qwen3.5-9B + LoRA", ("Qwen3.5-9B + LoRA SFT",)),
    ]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{ll cc}", r"\toprule", r" & Model & Subset-400 & Full (1{,}246) \\", r"\midrule"]
    for grp, label, keys in spec:
        cells = []
        for sname in ("subset400", "full"):
            vals = [rows.get((sname, k)) for k in keys]
            if all(v is None for v in vals):
                cells.append("--")
            else:
                cells.append(" / ".join(f3(v["auroc"]) if v else "--" for v in vals))
        L.append(f"{grp} & {label} & {cells[0]} & {cells[1]}" + r" \\")
    for k, lab in [("kNN-50 (embedding)", "Embedding kNN ($k{=}50$)"),
                   ("qwen3_8_27b few-shot retrieved", "Qwen3.8-27B + 4 retrieved shots"),
                   ("claude_sonnet_5_5 few-shot retrieved", "Sonnet + 4 retrieved shots")]:
        a, b = retr.get(("subset400", k)), retr.get(("full", k))
        if a or b:
            L.append(f"Retrieval & {lab} & {f3(a['auroc']) if a else '--'} & {f3(b['auroc']) if b else '--'}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Post-fire damage recognition} (AUROC, destroyed vs.\ not; 2024--25 test fires). "
          r"Subset-400: fixed stratified subset used for closed models. Fine-tuned and linear models are trained on "
          r"1{,}646 buildings from 2019--23 fires and selected on validation fires only.}",
          r"\label{tab:postfire}", r"\end{table}"]
    for k, m in [("Qwen3.5-4B + LoRA SFT (ours)", "pfSftFour"), ("Qwen3.5-9B + LoRA SFT", "pfSftNine"),
                 ("lp_post facebook/dinov2-base (post-hoc stricter ref)", "pfLp"), ("Qwen3.8-27B zero-shot | post", "pfQwenBig"),
                 ("Qwen3.5-4B base (same readout)", "pfBaseFour"), ("Qwen3.5-4B zero-shot (generative) | post", "pfGenFour")]:
        if ("full", k) in rows:
            macros[m] = f3(rows[("full", k)]["auroc"])
    for k, m in [("Claude Sonnet 5.5 zero-shot | post", "pfSonnet"), ("Claude Haiku 5.5 zero-shot | post", "pfHaiku")]:
        macros[m] = f3(rows[("subset400", k)]["auroc"])
    for p in comp["paired"]:
        if p["model"].startswith("Qwen3.5-9B") and p["reference"].startswith("lp_post"):
            macros["pfNineVsLp"] = sgn(p["delta"])
        if p["model"].startswith("Qwen3.5-4B") and p["reference"].startswith("lp_post"):
            macros["pfFourVsLp"] = sgn(p["delta"])
        if p["model"].startswith("Qwen3.5-4B") and p["reference"] == "Claude Haiku 5.5 zero-shot | post":
            macros["pfFourVsHaiku"] = sgn(p["delta"])
    return "\n".join(L)


def main() -> None:
    macros = {}
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "tab_prefire.tex").write_text(prefire_table(macros) + "\n")
    (PAPER / "tables" / "tab_postfire.tex").write_text(postfire_table(macros) + "\n")
    split = (EXP / "split_inflation.json")
    if split.exists():
        s = json.loads(split.read_text())
        for k, m in [("pre-fire | metadata HGB", "Meta"), ("post-fire | DINOv2 LP", "Post")]:
            macros[f"split{m}Event"] = f3(s[k]["event/time split"][0])
            macros[f"split{m}Random"] = f3(float(np.mean(s[k]["random split"])))
            macros[f"split{m}Within"] = f3(float(np.mean(s[k]["random within test events (same fires in train)"])))
    rv = EXP / "rerun_variance.json"
    if rv.exists():
        r = json.loads(rv.read_text())
        son = [abs(x["auroc_run1"] - x["auroc_run2"]) for x in r if x["model"] == "Sonnet"]
        hai = [abs(x["auroc_run1"] - x["auroc_run2"]) for x in r if x["model"] == "Haiku"]
        macros["rerunSonnetMax"] = f3(max(son))
        macros["rerunHaikuMax"] = f3(max(hai))
        macros["rerunAgreeMin"] = f"{min(x['label_agreement'] for x in r) * 100:.0f}"
        macros["rerunAgreeMax"] = f"{max(x['label_agreement'] for x in r) * 100:.0f}"
    rr = EXP / "postfire" / "retrieval" / "report.json"
    if rr.exists():
        rep = json.loads(rr.read_text())
        for t in rep["tests"]:
            key = {("R1", "qwen3_8_27b zero-shot"): "retrROne", ("R2", "qwen3_8_27b few-shot random"): "retrRTwo",
                   ("R3", "kNN-4 unweighted vote"): "retrRThree"}.get((t["test"], t["b"]))
            if key:
                macros[key] = sgn(t["delta"])
                macros[key + "Lo"] = sgn(t["ci95"][0])
                macros[key + "Hi"] = sgn(t["ci95"][1])
            if t["a"].startswith("claude") and t["b"] == "claude_sonnet_5_5 zero-shot":
                macros["retrSonnet"] = sgn(t["delta"])
        rows = {(r["set"], r["model"]): r["auroc"] for r in rep["rows"]}
        macros["retrQwenAuc"] = f3(rows[("subset400", "qwen3_8_27b few-shot retrieved")])
        macros["retrKnnFull"] = f3(rows[("full", "kNN-50 (embedding)")])
    # ---- mechanism analyses
    ca = EXP / "context_ablation.json"
    if ca.exists():
        c = json.loads(ca.read_text())
        for k, m in [("full", "Full"), ("center_19m", "Roof"), ("center_64m", "Centre"), ("masked_37m", "Masked"),
                     ("ring_only", "Ring")]:
            macros[f"ca{m}One"] = f3(c[k]["T1_auroc"])
            macros[f"ca{m}Two"] = f3(c[k]["T2_pair_acc"])
            macros[f"ca{m}Three"] = f3(c[k]["T3_auroc"])
    ra = EXP / "resolution_ablation.json"
    if ra.exists():
        r = json.loads(ra.read_text())
        macros["resZEighteenOne"] = f3(r["z18_center_75m"]["T1_auroc"])
        macros["resZNineteenOne"] = f3(r["z19_native_75m"]["T1_auroc"])
        macros["resZEighteenTwo"] = f3(r["z18_center_75m"]["T2_pair_acc"])
        macros["resZNineteenTwo"] = f3(r["z19_native_75m"]["T2_pair_acc"])
    qp = EXP / "postfire" / "qwen_vision_probe.json"
    if qp.exists():
        q = json.loads(qp.read_text())
        macros["probeQwenFour"] = f3(q["Qwen/Qwen3.5-4B"]["auroc"])
        macros["probeQwenNine"] = f3(q["Qwen/Qwen3.5-9B"]["auroc"])
    le = EXP / "postfire" / "lp_efficiency.json"
    if le.exists():
        l = json.loads(le.read_text())
        for n, m in [(100, "Hundred"), (400, "FourHundred"), (800, "EightHundred")]:
            macros[f"lpN{m}"] = f3(float(np.mean(l[str(n)])))
            d = EXP / "postfire" / f"sft_Qwen3.5-4B_n{n}_s0" / "log.json"
            if d.exists():
                macros[f"sftN{m}"] = f3(json.loads(d.read_text())["test"]["auroc"])
    fu = EXP / "prefire_fusion.md"
    if fu.exists():
        best = 0.0
        for line in fu.read_text().splitlines():
            if line.startswith("| test") or line.startswith("| val"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                best = max(best, max(float(x) for x in cells[2:8]))
        macros["fusionBest"] = f3(best)

    def vauc(run, ev):
        from sklearn.metrics import roc_auc_score
        p = EXP / "vlm_conditions" / run / "raw.jsonl"
        if not p.exists():
            return None
        bb = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["evidence"] == ev and (r["global_id"] not in bb or r.get("parse_ok")):
                bb[r["global_id"]] = r
        ok = [r for r in bb.values() if r.get("parse_ok")]
        return roc_auc_score([r["label"] == "Destroyed" for r in ok], [r["p_destroyed"] for r in ok]) if len(ok) > 100 else None
    for run, ev, m in [("pilot_sonnet_diag", "image_masked", "maskSonnetPre"), ("postfire_sonnet", "postfire_masked", "maskSonnetPost"),
                       ("postfire_sonnet", "postfire_image", "posctlSonnet"), ("pilot_qwen38_diag", "image_masked", "maskQwenPre"),
                       ("postfire_qwen38", "postfire_masked", "maskQwenPost"), ("postfire_qwen38", "postfire_image", "posctlQwen")]:
        v = vauc(run, ev)
        if v is not None:
            macros[m] = f3(v)
    lines = ["% auto-generated by scripts/paper/make_tables.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"[OK] {len(macros)} macros; tables written to paper/tables/")


if __name__ == "__main__":
    main()
