# Evidence Status Ledger

Last updated: 2026-10-19 (response to Review 1019). This file is the single source of truth for what the
project has *shown*, and it supersedes conflicting numbers in the README, the CVPR proposal and the internal
progress reports.

Evidence levels: **Empirical** (held-out, reproducible from logged artefacts) · **Dev-set** (real run, but on
the 50 cases used to build rules/targets) · **Demo** (UI or mechanism illustration, partly templated from labels)
· **Simulation** (no model trained or run) · **Planned**.

## 1. Claims ledger

| Claim in earlier docs | Level | What is actually true | Where |
|---|---|---|---|
| Web platform, case browser, live model calls, reward-audit UI | Engineering asset | Works; keep | `frontend/`, `backend/` |
| "Frontier VLMs fail at vision" on 50 cases (~88–100% false alarms) | Dev-set, **text-only** | Evaluator sends DINS post-fire attributes + one fixed Red Flag sentence; no image attached. FPR on survived homes: MiniMax-M3 88%, Haiku 88%, Sonnet 80%, Qwen 100% | `src/vlm_benchmark/evaluate_baselines.py` |
| "50 survived / destroyed single-family homes" | Dev-set | 44 SFR + 6 mobile homes; "Destroyed" = 24 Destroyed + 1 Major; "Survived" = 17 No Damage + 6 Affected + 2 Minor; 28/50 from Quail | `benchmark_results/dins_vlm_benchmark_50.json` |
| Ignition-locus attribution as a main task | Not supported | Specific ignition label on 0.32% of destroyed and 5.9% of survived residential DINS records | `output/data_inventory/dins_inventory.md` |
| GRPO "real training" / `trained_steps` | **Simulation** | No model loaded, no generation, no backward pass; fixed log-probs. Reward/loss wiring check only | `src/vlm_benchmark/qwen_training_pipeline.py` |
| SFT 82% / RLVR 94% | **Simulation** | Preset simulated policies | `src/vlm_benchmark/rlvr_post_training_runner.py` |
| Assimilated physics: accuracy up, time-of-ignition MAE 18.0 → 6.7 min | **Demo (synthetic)** | Default physics = constant `True`; "vanilla VLM" built from case id + true label; ToA set from true/predicted class; no per-case PDE/CA integration | `src/academic_models/benchmark_suite.py` (now returns `evidence_status: mechanism_demo_synthetic`) |
| Structured assessment, Zone 0 status, Bellwether probabilities per case | **Demo (label-conditioned)** | Templated from the true outcome. Never use as inputs/features | `src/vlm_benchmark/service.py` (`label_conditioned_demo_fields`) |
| "Verifiable physical reward" | Not reliable | 10/10 adversarial probes fail (negation, unknown labels, keyword spam, absolute-immunity physics) | `output/audits/reward_reliability_audit.json` |
| 100K multimodal research dataset | Planned | 132,522 DINS records downloaded; 90,005 residential with binary label; usable *paired, time-valid* subset still being measured | `output/data_inventory/` |
| "Pre-fire" imagery (WB_2020_R01) | Unverified for dev set | Release date ≠ capture date (e.g. a Quail case: release 2020-05, capture 2018-08-12). Old builder silently fell back to *current* imagery on tile errors (removed) | `src/vlm_benchmark/imagery_provenance.py` |
| Literature: DisasterM3 "NeurIPS 2024", FIRE-VLM "NeurIPS 2024, 34% lower tracking error" | Incorrect | DisasterM3 is NeurIPS 2025 D&B. FIRE-VLM (Webb et al.) is **WACV 2026 Workshops (CV4EO)** (earlier arXiv 2601.03449): PPO UAV fire-front tracking with CLIP potential-based reward shaping, evaluated only in a UE5/AirSim digital twin; reports up to 6× faster time-to-detection, no "34%" figure | Review 1019 §4; full text read 2026-10-22 |

## 2. New infrastructure (Weeks 1–2)

| Deliverable | Status | Artefact |
|---|---|---|
| Full paginated DINS download with page hashes | Done | `scripts/data/fetch_dins_full.py` → `/data/.../calfire_dins/full/` |
| Data-quality inventory (events, years, missingness-vs-outcome leakage, ignition-label coverage) | Done | `scripts/data/dins_inventory.py` → `output/data_inventory/` |
| Time-verified imagery: latest Wayback release before fire, per-location capture date, no fallback, SHA256 | Done | `src/vlm_benchmark/imagery_provenance.py` |
| Event-level splits; 50 legacy cases' 10 events quarantined as **dev**; forward-in-time test (2024+); 100 m spatial near-duplicate control | Done | `scripts/data/build_research_splits.py` → `output/splits/` |
| Pilot test set: 8 held-out events, balanced diagnostic (238) + natural-ratio (158) buildings, 396/400 imagery accepted | Done | `output/splits/pilot_test.jsonl`, `pilot_test_summary.md` |
| Controlled VLM harness (image actually attached; image / mismatched image / metadata / inspector × neutral / extreme prompt; full request+response logging, reported model id, image hash) | Done | `scripts/eval/run_vlm_conditions.py`, `scripts/eval/vlm_metrics.py` |
| Explicit metric names: FPR = FP/(FP+TN) vs false-alarm ratio = FP/(TP+FP); fire-event cluster-bootstrap CIs | Done | `scripts/eval/vlm_metrics.py`; legacy keys annotated |

## 3. Decisive experiments (Review 1019 §6)

| # | Experiment | Status | Result so far |
|---|---|---|---|
| 1 | Human determinability (expert labels from pre-fire inputs only, outcome hidden) | **Needs people** | Annotation protocol to draft; pilot images and attribute keys ready |
| 2 | Visual increment (meta vs image vs mismatched image) | Pilot run (Haiku) | Small, borderline: ΔAUROC +0.06 vs mismatched image (neutral prompt); vanishes under extreme prompt. §4 |
| 3 | Prompt bias (neutral vs fixed extreme weather) | Pilot run (Haiku) | Fixed Red Flag sentence: FPR +16 pts, AUROC down. Real event weather not yet built. §4 |
| 4 | Ground-truth attribute ceiling (leave-event-out) | Done | Clean observed-attribute subset: AUROC ≈ 0.71–0.72 (within-event 0.74–0.76); with "Unknown" as a category: 0.89–0.91, largely missingness leakage. `output/experiments/gt_attribute_ceiling.md` |
| 5 | Reward / evidence reliability | Probe audit done | 10/10 probes fail → do not optimise this reward. `output/audits/` |

### Simple baselines on the pilot test set (trained on 2019–2023 events)

From `output/experiments/tabular_baselines_pilot.md` (AUROC, 95% event-bootstrap CI):

| Features | Diagnostic (n=238) | Natural ratio (n=158) |
|---|---|---|
| Pre-fire metadata (type, year built, value), HGB | 0.625 [0.535, 0.724] | 0.553 [0.484, 0.645] |
| + post-fire inspector attributes (leaky oracle), logreg | 0.832 [0.732, 0.909] | 0.906 [0.849, 0.962] |

## 4. Pilot results: Exp #2 / #3

Model `claude-haiku-5-5` (API-reported id logged per call), 238-building balanced diagnostic set from 8 held-out
2024–2025 events, time-verified imagery, prompt `v1-2026-10-19`, API-default sampling (Claude models reject
`temperature`), one run. Parse failures 0–1.7% per condition after a max_tokens retry. Full table, per-event
results and every raw request/response: `output/experiments/vlm_conditions/pilot_haiku_diag/`.

| Evidence | Context | AUROC [95% CI] | FPR [95% CI] | mean p(D): Surv / Destr |
|---|---|---|---|---|
| meta | neutral | 0.519 [0.459, 0.591] | 46.6% [22.0, 71.7] | 0.477 / 0.489 |
| meta | extreme | 0.482 [0.402, 0.563] | 62.7% [44.9, 79.2] | 0.537 / 0.535 |
| image | neutral | 0.606 [0.544, 0.672] | 58.1% [39.8, 76.1] | 0.502 / 0.538 |
| image | extreme | 0.524 [0.420, 0.611] | 74.4% [61.0, 87.4] | 0.540 / 0.548 |
| image_mismatch | neutral | 0.543 [0.502, 0.592] | 61.0% [50.8, 71.9] | 0.513 / 0.527 |
| image_mismatch | extreme | 0.489 [0.394, 0.586] | 75.2% [60.3, 86.3] | 0.550 / 0.552 |
| inspector (leaky) | neutral | **0.395** [0.303, 0.476] | 66.9% [48.3, 82.5] | 0.524 / 0.454 |

Paired contrasts (same buildings, event-bootstrap 95% CI):

- **Visual increment (Exp #2):** under the neutral prompt, image − mismatched image ΔAUROC = +0.063 [+0.002, +0.130]
  and image − metadata ΔAUROC = +0.087 [+0.030, +0.143]. The increment is small and borderline. Under the extreme
  prompt it is not distinguishable from zero (+0.025 [−0.078, +0.139]).
- **Prompt bias (Exp #3):** adding the fixed Red Flag sentence raises FPR by +16 points in every evidence condition
  (meta +0.161 [+0.067, +0.250]; image +0.162 [+0.068, +0.244]). It lowers AUROC (image −0.083 [−0.157, −0.018]).
  So a large share of the legacy "Everything Burns" false-alarm rate comes from the experimental setup, not from
  the model's visual perception.
- **Attributes read by an LLM vs. a fitted model:** given the same post-fire inspector attributes, Haiku ranks
  *worse than chance* (AUROC 0.395). Logistic regression trained on 2019–2023 events reaches 0.832 on the same
  buildings. A likely explanation, still to be tested: the LLM applies a "combustible features → loss" prior,
  while DINS recording patterns run the other way (attributes of destroyed homes are more often Unknown, and
  survivors have their combustible features recorded). This alone can explain much of the legacy text-only result.
- **Absolute level:** the best Haiku condition (image, neutral: 0.606) is on par with a tabular model on pre-fire
  metadata (0.625 [0.535, 0.724]). No zero-shot VLM condition approaches the leaky-attribute ceiling.

Limits: one small model, one sampling run (run-to-run variance not yet measured), 8 events, balanced set only
(natural-ratio set not yet run), no real event weather yet.

Implications for next steps:
1. Neutral prompting is the default. A real per-event weather condition (RAWS/HRRR at fire start) replaces the fixed
   extreme sentence.
2. Before any training, repeat on ≥2 more models (Sonnet, Qwen if its endpoint accepts images) and on the natural set,
   with 2–3 repeats to estimate run variance.
3. The research question "what can be reliably determined from pre-fire imagery" is supported. The visual signal
   exists but is weak at 0.3–0.6 m resolution, so Exp #1 (human determinability) is now the critical path.

## 4b. VLM role study (2026-10-20) — see `docs/VLM_ROLE_AND_BASELINE_DESIGN.md`

| Finding | Evidence |
|---|---|
| Building-level ceiling is low: clean inspector attributes reach only 0.62–0.65 near-neighbour pair accuracy (0.89 with missingness leakage) | `output/experiments/exposure_control.md` |
| Perception is adequate: post-fire positive control AUROC Sonnet 0.959, Haiku 0.924, Qwen 0.846 | `vlm_conditions/postfire_*` |
| Frozen DINOv2/SigLIP + LR beats every VLM direct judgment (0.67–0.71 vs 0.52–0.61) but is at chance on pairs (0.52–0.56): it captures location/exposure, not building vulnerability | `embedding_baseline/` |
| Pairwise (exposure-controlled): Sonnet 0.620, MiniMax 0.605, Qwen 0.589, Haiku 0.571 vs tabular metadata 0.604; Sonnet image − meta +0.039 [+0.004, +0.085] | `vlm_pairs/`, `pairs_reference.md` |
| Qwen ignores the image in direct judgment (image − mismatched −0.004) | `vlm_conditions/pilot_qwen38_diag` |
| VLM-as-sensor (Qwen extraction → LR head) fails: pairs 0.523; fine-grained attributes not rotation-stable | `vlm_extraction/extract_qwen38/analysis.md` |
| DINS points are not a valid spacing reference; Overture 2026 is post-fire | design doc §4 |

## 4c. Graded conclusions (revised 2026-10-20 after pre-registered confirmation on untouched val events)

Plan and code were committed before any run: `docs/CONFIRMATORY_PLAN.md` (commit fb92a36); results in
`output/experiments/confirmatory/results.md`. Deviation: Holm-correction code bug fixed after the first
run; no analysis rule changed.

**Level 1 — paper-ready (own data, independent lines agree, replicated where tested)**

| Claim | Evidence |
|---|---|
| Building state is perceivable in these crops (post-fire control) | Sonnet 0.959 [0.948, 0.970], Haiku 0.924, Qwen 0.846 (106/118 parsed); 118/238 with a capture ≤365 d after the fire |
| Open models lag in *perception*, not only reasoning | post-fire AUROC: Qwen 0.846 [0.791, 0.917] vs Sonnet 0.959 (CIs do not overlap) |
| VLM direct judgment uses the image (own vs mismatched) — but its absolute level stays below a frozen-embedding head | Sonnet: test +0.068, **confirmed** on val +0.155 [+0.074, +0.234] (Holm p=0.0015); Qwen val +0.094 [+0.002, +0.177] |
| DINS attribute *values* carry ~0.63 pair accuracy; ~0.88 comes from missingness | same population: missingness indicators only 0.883, random-imputed values 0.626, clean subset 0.62–0.65 (`missingness_ablation.md`, `exposure_control.md`) |
| Frozen-image embeddings rank by location more than by building | AUROC 0.665–0.714 (test), 0.672 (val) vs pairs 0.52–0.56 (test), 0.571 (val) |
| Zero-shot VLM direct judgment does not beat a frozen-embedding linear head | VLMs 0.52–0.61 (test), Sonnet 0.641 (val) vs DINOv2 head 0.665–0.714 (test), 0.672 (val) |
| VLM-as-sensor fails *for Qwen3.8-27B at 0.3–0.6 m with this schema and 971 training buildings* | rotation agreement 58–66% on fine attributes; head pair acc 0.523 (Sonnet extraction + head untested) |
| Data-engineering facts | Wayback release date ≠ capture date (median gap 585 d); old builder *had* a silent fallback to current imagery (removed; never verified to have fired); Overture 2026 is post-fire |

| Post-fire: a 4B open VLM with LoRA SFT on 1,646 train-event buildings beats zero-shot Haiku/Sonnet and a 27B open model on unseen 2024+ fires, **but only ties a frozen DINOv2-base + LR** | SFT 0.892 (full test); vs Haiku post +0.095 [+0.046, +0.147], vs Sonnet post +0.065 [+0.021, +0.106], vs Qwen3.8-27B +0.192; vs frozen LP +0.011 [−0.007, +0.028] (`postfire/comparison.md`, pre-registered in `POSTFIRE_TASK_PLAN.md`) |
| Readout matters: same 4B base, generated number 0.644 vs first-token probability 0.762 | `postfire/comparison.md` |
| Scale (9B vs 4B, same recipe): small, not significant gain (0.903 vs 0.892, Δ +0.011 [−0.003, +0.023]); 9B base is *worse* than 4B base before SFT (0.709 vs 0.762) | `postfire/comparison.md` |
| Output-form value over a frozen linear head: no selective-prediction advantage (AURC Δ ≈ 0); 5-level grading advantage borderline (9B QWK lower bound +0.0001; 4B not met); minor/major levels unrecognisable for every model | `postfire/output_value.md` |

| Evaluation protocol inflates pre-fire risk scores far more than post-fire recognition (exploratory) | random building split vs event/time split: pre-fire metadata 0.712 vs 0.551 (+0.161; same-fire training alone 0.697); post-fire DINOv2 LP 0.933 vs 0.881 (same-fire training 0.888 ≈ no leak) (`split_inflation.md`) |

**Level 2 — real numbers, interpretation or generality open**

| Claim | Status |
|---|---|
| VLM pairwise judgment beats metadata under exposure control | **not supported**: Sonnet test 0.620 / val 0.555 (H2 Holm p=0.053); Qwen test 0.589 / val 0.599; pooled both ≈0.593 vs tabular metadata 0.59–0.60; model ranking flips between event sets |
| Image adds over metadata in pairs | test +0.039 [+0.004, +0.085]; val +0.024 [−0.038, +0.090] (H1 had 37% power) |
| ~~Qwen ignores the pre-fire image~~ **not supported** | test −0.004 [−0.069, +0.079] but val +0.094 [+0.002, +0.177]; Qwen val AUROC 0.639 ≈ Sonnet 0.641. The open-model gap is established only for post-fire perception |
| Haiku reverses direction when reading inspector attributes | test 0.395; **not replicated** on val (0.517). Stable piece: Haiku treats "Unknown" as *lower* risk (corr −0.26 test, −0.35 val); the reversal appears only where missingness strongly tracks loss (corr +0.62 test vs +0.10 val) |
| Fixed extreme-weather prompt raises FPR (~+16 pp) | Haiku, single run; Qwen direction consistent on 26 cases only |
| Density rule learned on 2019–23 fires does not transfer to 2024–25 | 0.391, from DINS-point density (itself incomplete), test dominated by Eaton/Palisades — a transfer failure, not a test of LLM common sense |
| DINS points miss structures | inferred from Sonnet's systematically shorter distances; not checked against footprints |
| Mechanism of the low ceiling (decisive features invisible from nadir; ember stochasticity; suppression) | hypothesis consistent with data, not tested |

**Level 3 — external literature numbers**: verify each against the original paper before citing. Verified from full text (2026-10-22):

| Paper | What it does | Evaluation / caveats | Relation to us |
|---|---|---|---|
| Chen et al., *A Simulation-Grounded Agentic VLM Framework for Wildfire Monitoring and Reporting*, arXiv 2610.02451 (under review) | SimFire (Rothermel, LANDFIRE) → Blender proxy → controllable video generation → simulator-labelled video memory; training-free multi-agent VLM retrieval + reporting | Held-out **generated** episodes: 51.5% exact 4-tag vs 22.6% direct VLM; 77.3% on 6 simulator fields; cross-generator drops to 41–44%; real UAV only for smoke/fire detection and visible area; authors note metadata-conditioned prompts may leak tag cues | Sim-to-sim; its gain comes mostly from retrieval of labelled exemplars (direct-visual agent alone 66.8%). We test the retrieval idea on real outcomes (POSTFIRE_TASK_PLAN §10) |
| Ayanzadeh et al., *WildfireVLM*, arXiv 2602.13305 | YOLOv12 fire/smoke detection on 3,771 Landsat/GOES patches + GPT-4o / Claude 3.5 risk narratives | Random 70/15/15 patch split; narratives scored 1–10 by an LLM judge (7.03 vs 6.16), no ground-truth risk labels | "Detect then narrate"; contrast for protocol: random splits and LLM-as-judge (cf. `split_inflation.md`) |
| Webb et al., *FIRE-VLM*, WACV 2026 Workshops (CV4EO) | PPO UAV control in a DEM/LANDFIRE/CAWFE digital twin; CLIP ViT-B/32 prompt similarity as potential-based reward shaping + quadrant direction bonus | Simulation only; ablations only (no external baselines), N=5 runs; prompt wording changes the reward | Principled use of a noisy VLM signal (shaping, not truth); a control task, not ours |

**Level 4 — reasoning only, no data**: post-fire SFT closes Qwen's gap; street-level view adds to nadir; calibration/abstention as differentiator; decision-time advice.

## 5. Rules going forward

1. The 50 legacy cases and their 10 events are **dev only**. Never report them as test results.
2. Model inputs may use only `inputs.prefire` and imagery with `imagery.accepted == true`. DINS inspector
   attributes are post-fire; use them only in explicitly labelled oracle/ceiling conditions.
3. Every reported number must point to a logged artefact (raw requests/responses, image hashes, reported model id).
4. Report FPR and false-alarm ratio by those names; report per-event results and event-cluster intervals.
5. Ignition-location accuracy only on records with a specific label, with coverage reported; Unknown ≠ negative.
6. No RL until (a) the reward passes the probe audit and agrees with independent expert scoring, and
   (b) SFT / constrained decoding leave a stable, verifiable error that RL is compared against at equal compute.
