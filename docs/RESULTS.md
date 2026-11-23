# Results

What the experiments in this repository show, with the numbers and where they come from. Every number is
computed from logged model outputs under `output/` (kept locally, not in the repository; each script below
regenerates its part). Last updated 2026-11-23.

**In short.**

- **After a fire, damage is easy to see.** Zero-shot Claude Sonnet reaches AUROC .936 from pre- and post-fire photos.
  A 4B open model fine-tuned on 1,646 homes reaches .892, but a frozen DINOv2 encoder with logistic regression does
  about as well (.881).
- **Before a fire, the building-level signal in overhead photos is weak.** Shown two neighbouring homes from the
  same fire, one destroyed, the best zero-shot VLM picks the right one 62% of the time on the test fires and 55.5%
  on the replication fires. That is about what year built and structure type give (59–60%).
- **The evaluation protocol matters.** Splitting by building instead of by fire event inflates pre-fire scores by
  .16 AUROC.

## Setup

| | |
|---|---|
| Labels | CAL FIRE Damage Inspection (DINS), 132,522 records. Residential binary label: Destroyed + Major (>25%) vs. No Damage / Affected / Minor. |
| Splits | By fire event and forward in time: training and tuning on 2019–23 fires, test on 2024–25 fires, a separate set of 2019–23 validation fires for the replication, and a 100 m buffer between splits. An earlier 50-case development set is excluded from every result. |
| Imagery | Esri World Imagery Wayback crops (zoom 18, 512 px, about 150 m across). The capture date at each building is read from Wayback metadata: a release's date and its capture date differ by a median of 585 days. Pre-fire = newest capture before the fire start; post-fire = earliest capture within 365 days after. Crops whose date cannot be verified are dropped, never replaced. |
| Models | Claude Sonnet 5.5 and Haiku 5.5, Qwen3.8-27B, MiniMax-M3 (zero-shot); Qwen3.5-4B / 9B, zero-shot and LoRA fine-tuned; frozen DINOv2 / SigLIP encoders with a linear probe; gradient-boosted and logistic models on building metadata. |
| Scoring | AUROC for single buildings, accuracy for pairs. 95% intervals are bootstrap intervals clustered by fire event. Answers that cannot be parsed count as p = 0.5. Every request and response is logged with the model id the API reported and the image hash. |

## 1. After the fire: was this home destroyed? (T3)

Test: 1,246 buildings from 83 fires in 2024–25 (Claude models on a fixed 400-building subset).

| Model | Input | AUROC |
|---|---|---|
| Claude Sonnet 5.5, zero-shot | pre + post images | **.936** |
| Claude Haiku 5.5, zero-shot | pre + post images | .918 |
| Qwen3.5-9B + LoRA (trained on 2019–23 fires) | post-fire image | .903 |
| Qwen3.5-4B + LoRA (trained on 2019–23 fires) | post-fire image | .892 |
| Claude Sonnet 5.5, zero-shot | post-fire image | .892 |
| DINOv2-base, frozen, linear probe | pre + post / post only | .890 / .881 |
| Claude Haiku 5.5, zero-shot | post-fire image | .862 |
| Qwen3.8-27B, zero-shot | pre + post / post only | .742 / .701 |
| Qwen3.5-4B, zero-shot | post-fire image | .644 |

- **Fine-tuning closes the open-model gap.** With LoRA on 1,646 buildings the 4B model gains .130 over the same model
  untuned and beats zero-shot Haiku (+.095 [+.045, +.145]) and Sonnet (+.065 [+.016, +.104]) on the same post-fire
  images.
- **But it only ties a cheap supervised image model.** Against the frozen DINOv2-base probe the difference is +.011
  [−.008, +.029].
- **How the score is read out matters.** The same 4B base scores .644 when it writes a number and .762 when the
  probability of its first answer token is used.
- **No gain from:** a larger model (9B vs 4B: +.011 [−.003, +.023]); four retrieved labelled examples in the prompt
  (Qwen +.019 [−.023, +.061], Sonnet −.012 [−.069, +.038]); or a graded output (the five DINS damage levels are hard
  for every model, macro-F1 .36 at best).
- **Hardest fires:** the 2025 urban conflagrations (Eaton .805, Palisades .774 for the 4B LoRA model).
- **Outside California:** on the 2021 Marshall Fire in Colorado (800 buildings, BRIGHT labels, models trained on
  California only) every model scores .98, so this set does not separate the methods.

Rules for starting fine-tuning and for judging it were fixed before any result:
[`POSTFIRE_TASK_PLAN.md`](POSTFIRE_TASK_PLAN.md) (in Chinese). Scripts: `scripts/data/build_postfire_dataset.py`,
`scripts/eval/run_vlm_conditions.py --evidence postfire_image`, `scripts/train/sft_postfire_qwen.py`,
`scripts/eval/postfire_compare.py`, `postfire_output_value.py`, `run_retrieval_fewshot.py`,
`postfire_external_eval.py`.

## 2. Before the fire: which of two neighbours burned? (T2)

Pairs are two homes 15–100 m apart in the same fire, exactly one destroyed. Fire, weather and timing are the same
for both, so the pair tests what is visible on the building and its yard. Test: 403 pairs from 13 fires in 2024–25.

| Model or baseline | Input | Pair accuracy |
|---|---|---|
| Claude Sonnet 5.5 | two pre-fire images | .620 |
| Metadata model (structure type, year built), fitted on 2019–23 fires | metadata | .604 |
| MiniMax-M3 | two pre-fire images | .603 |
| Qwen3.8-27B | two pre-fire images | .588 |
| Claude Sonnet 5.5 | metadata only | .578 |
| Claude Haiku 5.5 | two pre-fire images | .571 |
| Defensible-space measurements (segmented vegetation and building spacing around each home) | pre-fire image | .556 |
| Frozen DINOv2 / SigLIP ranker | pre-fire image | .52–.56 |

- **Replication on untouched fires.** On 289 pairs from validation fires, Sonnet scores .555 [.500, .611]. The
  pre-registered test that it beats chance does not pass after correction (Holm p = .053). Qwen3.8-27B scores .599.
- **Pooled over 692 pairs**, Sonnet and Qwen both reach about .593, the same as the metadata model (.59–.60). Under
  exposure control, a zero-shot VLM tells neighbours apart about as well as knowing when each house was built. Which
  of the two models comes out ahead flips between the two sets of fires.
- **The ceiling is low even with inspection data.** The attribute values CAL FIRE inspectors record (roof, siding,
  vents, ...) reach only .63–.65 pair accuracy. Whether an attribute was recorded at all reaches .88, because
  destroyed homes are more often recorded as "Unknown": a leak, not a signal.

Plan and frozen analysis code: [`CONFIRMATORY_PLAN.md`](CONFIRMATORY_PLAN.md) (in Chinese). Scripts:
`scripts/eval/run_vlm_pairs.py`, `confirmatory_analysis.py`, `defensible_space_eval.py`.

## 3. Before the fire: one building at a time (T1)

- Zero-shot VLMs score AUROC .52–.61 on the test fires and .641 (Sonnet) on the replication fires.
- They do use the image: replacing it with another building's photo drops Sonnet by .155 [+.074, +.234] (confirmed,
  Holm p = .0015).
- A frozen DINOv2 encoder with a linear probe still scores higher (.665–.714 test, .672 replication). Its pair accuracy
  is near chance (.52–.57), so it ranks locations, not buildings.
- A fixed "Red Flag Warning" sentence in the prompt raises Haiku's false-positive rate by about 16 points and lowers
  its AUROC. Prompts here state no weather unless it is the real weather of the event.

Scripts: `scripts/eval/run_vlm_conditions.py`, `vlm_metrics.py`, `embedding_baseline.py`.

## 4. How the split changes the answer

The same models scored with a random building split instead of a split by fire event and time:

| | Random building split | Event / time split |
|---|---|---|
| Pre-fire metadata model, AUROC | .712 | .551 |
| Post-fire DINOv2 probe, AUROC | .933 | .881 |

Most of the pre-fire inflation comes from training on other buildings of the same fire (.697 with same-fire training
alone). Post-fire recognition barely changes. Script: `scripts/eval/split_inflation.py`.

## 5. Exploratory pilots (not pre-registered)

- **Community scale.** About 500 m cells with at least 10 inspected homes; 377 test cells from 22 fires; the label is
  whether at least half the homes were lost. Vegetation within 1 km (ESA WorldCover) scores .79 AUROC on wildland-edge
  fires but reverses in the 2025 urban fires, and zero-shot VLM reasoning inherits the reversal (Sonnet .45, Qwen .51
  overall). Scripts: `scripts/data/build_community_cells.py`, `scripts/eval/run_community_vlm.py`, `community_eval.py`.
- **Evacuation-decision replay.** From the fire perimeters known at each decision time (NIFC archive) and ERA5 wind,
  which of two communities at the same distance will the fire reach? Sonnet with a map scores .715 (+.064 over
  text only); distance and a wind-adjusted rule score .70–.72. Scripts: `scripts/data/build_evac_replay.py`,
  `scripts/eval/evac_baselines.py`, `run_evac_vlm.py`.

## Rules followed

1. Model inputs use only pre-fire information and imagery with a verified capture date. Post-fire inspector
   attributes appear only in conditions labelled as ceilings.
2. Every reported number points to logged outputs (raw requests and responses, image hashes, reported model id).
3. Hypotheses, metrics and decision rules for the replication and the post-fire task were committed before the runs
   they judge; deviations are listed in those documents.
