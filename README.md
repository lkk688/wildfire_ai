# Wildfire AI: verifiable evaluation of vision-language models for structure loss

Research code for evaluating whether vision-language models (VLMs) can tell, from aerial imagery, which homes
**will be** destroyed in a wildfire (pre-fire) and which **were** destroyed (post-fire), against real outcomes
from CAL FIRE damage inspections (DINS).

> **Read this first.** This repository contains two kinds of code:
>
> 1. **The research pipeline** (`scripts/`, `src/vlm_benchmark/imagery_provenance.py`, `paper/`). Every number it
>    produces is computed from real records and real model outputs. What is and is not established is tracked
>    in [`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md), the single source of truth for results.
> 2. **A legacy demo application** (`backend/`, `frontend/`, and parts of `src/`), built before 2026-10-19.
>    **Much of its displayed content is synthetic, hard-coded, or generated from the true outcome label.** It must not
>    be used for decisions or presented as results. See [Legacy demo application](#legacy-demo-application) below.

## Research pipeline

**Data and splits.**
- 132,522 DINS records; the residential binary label is Destroyed + Major vs. No Damage / Affected / Minor.
- Splits are by fire event and forward in time: train on 2019–23 fires, test on 2024–25 fires.
- The 50-case legacy development set is quarantined, and buildings near another split's buildings are excluded
  (100 m buffer).

**Imagery provenance.** [`src/vlm_benchmark/imagery_provenance.py`](src/vlm_benchmark/imagery_provenance.py)
fetches Esri World Imagery Wayback crops and records each crop's verified per-location **capture date**, not just its
release date.
- A pre-fire crop must be captured before the fire.
- A post-fire crop is the earliest capture within 365 days after the fire.
- There is no fallback to current imagery: if the capture date cannot be verified, the sample is rejected.

**Tasks.**

| Task | Question | Main script |
|---|---|---|
| T1 | Pre-fire, single building: P(destroyed) | `scripts/eval/run_vlm_conditions.py` |
| T2 | Pre-fire, same-fire neighbour pairs 15–100 m apart (exposure-controlled): which one was destroyed? | `scripts/eval/run_vlm_pairs.py` |
| T3 | Post-fire damage recognition | `scripts/eval/run_vlm_conditions.py --evidence postfire_image`, `scripts/train/sft_postfire_qwen.py` |
| Replication | Pre-registered confirmatory tests on untouched validation fires | `scripts/eval/confirmatory_analysis.py`, [`docs/CONFIRMATORY_PLAN.md`](docs/CONFIRMATORY_PLAN.md) |

**Baselines and analyses.** All are in `scripts/eval/`:
- tabular models
- frozen DINOv2/SigLIP probes and a pair ranker
- fine-tuned DINOv2
- masking, resolution and split-inflation analyses
- missingness leakage
- retrieval

**Paper tables and figures.** These are generated only from `output/` by `scripts/paper/make_tables.py` and
`make_figures.py`. VLM answers that fail to parse count as p = 0.5 (intention-to-treat).

**Exploratory pilots (2026-10-22, feasibility only).**
- *Community-scale risk:* `scripts/data/build_community_cells.py`, `worldcover_features.py`.
- *Evacuation-decision replay:* archived NIFC perimeters, with no future information at decision time.
  See `scripts/data/evac_perimeter_inventory.py`, `build_evac_replay.py`, `ms_building_cells.py`,
  `render_evac_maps.py`, `scripts/eval/evac_baselines.py` and `run_evac_vlm.py`.

`output/` is git-ignored. API keys are read from the environment (`CLAUDE_API_KEY` / `ANTHROPIC_API_KEY`,
`QWEN_REMOTE_KEY` with `TB_REMOTE_QWEN_URL`, `MINIMAX_API_KEY`) and are never stored in the repository.

### Documentation

| Document | Content |
|---|---|
| [`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md) | Claims ledger: the evidence level of every result |
| [`docs/CONFIRMATORY_PLAN.md`](docs/CONFIRMATORY_PLAN.md) | Pre-registration and results of the replication |
| [`docs/POSTFIRE_TASK_PLAN.md`](docs/POSTFIRE_TASK_PLAN.md) | Pre-registration and results of the post-fire task |
| [`docs/VLM_ROLE_AND_BASELINE_DESIGN.md`](docs/VLM_ROLE_AND_BASELINE_DESIGN.md) | How VLMs are used, and which baselines they are compared with |
| [`docs/RELATED_WORK_WIRC.md`](docs/RELATED_WORK_WIRC.md) | Related work (SJSU WIRC survey) |
| [`docs/archive/`](docs/archive/) | **Archived** pre-10-19 documents, each headed by a list of known errors. Not a source of results. |

### Data sources and licences

| Source | Terms |
|---|---|
| CAL FIRE DINS | Public records |
| Esri World Imagery / Wayback | Esri terms of use; crops are not redistributed |
| NIFC Operational Data Archive | Public domain |
| Microsoft US Building Footprints | ODbL |
| ESA WorldCover 2021 | CC BY 4.0 |
| ERA5 via Open-Meteo | CC BY 4.0 |
| Bellwether rasters | Shared by the partner; not redistributed |

## Legacy demo application

`backend/` (FastAPI) and `frontend/` (Next.js) implement a GIS dashboard, a VLM "workstation" and an AI copilot.
An audit on 2026-10-22 found that, in their current state, many displayed values are not real:

- **Label-conditioned values (generated from the true DINS outcome).** These are shown as model outputs:
  - "Bellwether" probabilities: 0.0342 if destroyed, else 0.0125 (5-year 0.158 / 0.058)
  - the Zone 0 and defensible-space status
  - the insurance and Verisk figures
  - the academic-simulation numbers
  - the expert "forensic" diagnoses
  - the `structured_assessment` returned by `/api/vlm/predict`, even when the model call fails
  - the RLVR / SFT / "WildfireReason" benchmark predictions (`rlvr_post_training_runner.py`)
  - the SFT "gold" reasoning traces (`cot_dataset_generator.py`)

  Real Bellwether rasters exist only for the partner pilot area (`/api/layers/bellwether`, `/api/risk-factors`).
- **Synthetic data labelled as live:**
  - the CAL FIRE incidents, including fictitious "active" incidents
  - FIRMS hotspots, CDEC/RAWS fuel moisture, HRRR weather
  - the "USGS 3DEP LiDAR" slope (a sine pattern)
  - the "Microsoft Building Footprints" layer (an 8×8 grid with fake APNs)
  - the FHSZ overlay (concentric circles)
  - the corridor burn probability (a constant)
  - the land cover (chosen by hashing the coordinates)

  The copilot's tools return these same values.
- **Hard-coded results in the UI:** the "Ours" tables (82% assimilation, 94% RLVR), the fallback values (e.g. '3.42%',
  '$3,250,000') and the venue badges. The "Commercial SOTA" and "NDWS" benchmark rows were constants and a hand-drawn
  ellipse.
- **The 50-case development set is confounded:** all 25 survived cases come from one fire, and its evaluator prompt
  includes the fire name. Its scores (text only, no image) say nothing about vision.
- **`src/README.md` and `src/walkthrough.md`** describe this demo and have not been re-audited line by line.

Until each component is rebuilt on real outputs, treat the whole demo as an interface mock-up. The provider-switching
endpoint is disabled unless `WILDFIRE_DEMO_ALLOW_PROVIDER_SWITCH=1` is set, and it never accepts a custom `base_url`
from the network.

## Quick start (research pipeline)

```bash
python scripts/data/build_research_splits.py              # event-level, forward-in-time splits
python scripts/eval/run_vlm_conditions.py --help          # zero-shot VLM conditions (T1 / T3)
python scripts/eval/run_vlm_pairs.py --help               # exposure-controlled pairs (T2)
python scripts/paper/make_tables.py && python scripts/paper/make_figures.py
```
