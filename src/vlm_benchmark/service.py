"""
Wildfire VLM Benchmark Service
Provides data loading, result aggregation, expert forensic diagnosis, and live inference
for the WildfireRisk Vision & Reasoning frontend application.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.vlm_benchmark.evaluate_baselines import DINSBaselineEvaluator

BENCHMARK_DIR = Path("/Developer/wildfire_ai/SBFire_SJSU-20260730T030001Z-1-001/benchmark_results")
IMAGES_DIR = BENCHMARK_DIR / "images"
BENCHMARK_JSON = BENCHMARK_DIR / "dins_vlm_benchmark_50.json"


# Fields that earlier versions generated from the true outcome label. They are now always None.
LABEL_CONDITIONED_DEMO_FIELDS = [
    "bellwether_prediction", "academic_simulations", "insurance_actuarial",
    "expert_diagnosis", "structured_assessment", "incident_forensics",
]


class VLMBenchmarkService:
    def __init__(self, benchmark_dir: Optional[Path] = None):
        self.benchmark_dir = benchmark_dir or BENCHMARK_DIR
        self.images_dir = self.benchmark_dir / "images"
        self.benchmark_json = self.benchmark_dir / "dins_vlm_benchmark_50.json"
        self._cached_catalog = None
        self._cached_results = None
        self._dins_lookup = None

    def _load_dins_lookup(self) -> Dict[int, Any]:
        """Load full DINS GeoJSON lookup table for assessor and structural enrichment."""
        if self._dins_lookup is not None:
            return self._dins_lookup
        dins_file = Path("/data/wildfire_benchmark_data/calfire_dins/calfire_dins_structures.geojson")
        lookup = {}
        if dins_file.exists():
            try:
                with open(dins_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for feat in data.get("features", []):
                        props = feat.get("properties", {})
                        oid = props.get("OBJECTID")
                        if oid is not None:
                            lookup[oid] = props
            except Exception as e:
                print(f"Error loading DINS geojson: {e}")
        self._dins_lookup = lookup
        return lookup

    def get_catalog(self) -> Dict[str, Any]:
        """Load benchmark cases catalog with multi-epoch imagery, street view, and expert forensic diagnosis."""
        if not self.benchmark_json.exists():
            return {"total": 0, "items": []}

        with open(self.benchmark_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        dins_lookup = self._load_dins_lookup()
        items = data.get("items", [])
        enriched_items = []
        for item in items:
            enriched = dict(item)
            case_id = str(item["id"])
            cid_int = int(item["id"]) if str(item["id"]).isdigit() else None
            dins_props = dins_lookup.get(cid_int, {}) if cid_int is not None else {}

            enriched["image_url"] = f"/api/vlm/benchmark/image/{case_id}?epoch=prefire"
            enriched["has_image"] = (self.images_dir / f"case_{case_id}.jpg").exists()

            # APN & Assessor Records
            apn = dins_props.get("APN") or item.get("apn") or "Unknown"
            year_built = item.get("year_built") or dins_props.get("YEARBUILT")
            assessed_val = item.get("assessed_value") or dins_props.get("ASSESSEDIMPROVEDVALUE")

            # Chapter 7A Compliance Evaluation
            try:
                year_num = int(year_built) if year_built is not None else None
            except (ValueError, TypeError):
                year_num = None

            if year_num is not None:
                if year_num >= 2008:
                    chapter_7a = {
                        "is_compliant": True,
                        "code_status": "Post-Chapter 7A (2008+ Mandatory Code)",
                        "badge_type": "compliant",
                        "cutoff_year": 2008,
                        "standard_summary": "Mandated compliance with California CBC Title 24 Chapter 7A strict ember-resistant standards (Class A roof, <=1/8-inch non-combustible mesh vents, dual-pane tempered glass, non-combustible siding, boxed eaves).",
                        "regulatory_basis": "California Building Code Title 24, Part 2, Chapter 7A (Mandatory in WUI Zones post-2008)",
                    }
                else:
                    chapter_7a = {
                        "is_compliant": False,
                        "code_status": "Pre-Chapter 7A (1900-2007 Historical Standard)",
                        "badge_type": "non_compliant",
                        "cutoff_year": 2008,
                        "standard_summary": "Constructed prior to mandatory Chapter 7A enforcement; vulnerable to attic ember intrusion through legacy oversized mesh (>1/8-inch), unenclosed eaves, and combustible siding seams.",
                        "regulatory_basis": "Pre-Standard Construction (Built before 2008 WUI code enactment)",
                    }
            else:
                chapter_7a = {
                    "is_compliant": None,
                    "code_status": "Year Built Unrecorded",
                    "badge_type": "unknown",
                    "cutoff_year": 2008,
                    "standard_summary": "Construction year unrecorded in county assessor records; visual inspection of roof and facade texture required.",
                    "regulatory_basis": "Assessor Year Built Unrecorded",
                }


            enriched["assessor_record"] = {
                "apn": apn,
                "year_built": year_built,
                "assessed_improved_value": assessed_val,
                "assessed_value_display": f"${assessed_val:,.0f}" if isinstance(assessed_val, (int, float)) and assessed_val > 0 else "N/A",
                "chapter_7a": chapter_7a,
            }

            # Multi-temporal imagery links
            has_streetview = (self.images_dir / f"case_{case_id}_streetview.jpg").exists()
            enriched["multi_epoch_images"] = {
                "prefire": {
                    "url": f"/api/vlm/benchmark/image/{case_id}?epoch=prefire",
                    "label": "Pre-fire image (Wayback release WB_2020_R01)",
                    "date": "capture date not verified",
                    "role": "VLM physical deduction input with intact structural envelope",
                    "badge": "Pre-Fire Baseline",
                    "exists": (self.images_dir / f"case_{case_id}_prefire.jpg").exists() or (self.images_dir / f"case_{case_id}.jpg").exists(),
                },
                "postfire": {
                    "url": f"/api/vlm/benchmark/image/{case_id}?epoch=postfire",
                    "label": "Post-fire image",
                    "date": "capture date not verified",
                    "role": "CAL FIRE post-fire damage assessment ground truth evidence",
                    "badge": "Post-fire image",
                    "exists": (self.images_dir / f"case_{case_id}_postfire.jpg").exists(),
                },
                "current": {
                    "url": f"/api/vlm/benchmark/image/{case_id}?epoch=current",
                    "label": "Current imagery",
                    "date": "capture date not verified",
                    "role": "Current parcel recovery and vegetative regrowth status",
                    "badge": "Latest Satellite",
                    "exists": (self.images_dir / f"case_{case_id}_current.jpg").exists(),
                },
                "streetview": {
                    "url": f"/api/vlm/benchmark/image/{case_id}?epoch=streetview",
                    "label": "Google Street View Micro Ground Facade",
                    "date": "Ground Level Micro Perspective",
                    "role": "Micro facade envelope, roof eaves, fence junctions, and Zone 0 defensible buffer",
                    "badge": "Micro Ground Facade",
                    "exists": has_streetview,
                },
            }

            # Micro Street View Metadata & Facade Annotations
            lat, lon = item.get("latitude", 0), item.get("longitude", 0)
            if has_streetview:
                provenance = "street_level_photo_unverified_date"
                provenance_badge = "Street-level photo (date unverified)"
                provenance_label = "Street-level photograph; capture date and pre/post-fire status not verified"
                explanation = "Street-level photograph near the structure. Its capture date is not recorded, so it may show the building before or after the fire and must not be treated as ground truth."
            else:
                provenance = "no_street_level_photo"
                provenance_badge = "No street-level photo"
                provenance_label = "No street-level photograph available for this case"
                explanation = "No street-level photograph is stored for this case."

            enriched["streetview"] = {
                "url": f"/api/vlm/benchmark/image/{case_id}?epoch=streetview",
                "has_image": has_streetview,
                "provenance": provenance,
                "provenance_badge": provenance_badge,
                "provenance_label": provenance_label,
                "explanation": explanation,
                "google_maps_url": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}",
                "google_coords_url": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
                "facade_annotations": [],  # hand-written annotations removed (audit 2026-10-22)
            }

            # Multi-Paradigm Predictive Models & Insurance Actuarial Benchmarks
            # Audit 2026-10-22: the Bellwether / academic-simulation / insurance / forensic / expert-diagnosis /
            # structured-assessment fields used to be generated from the true outcome label. They are removed; the
            # UI shows "—". Real Bellwether values exist only inside the partner raster (/api/layers/bellwether).
            for k in LABEL_CONDITIONED_DEMO_FIELDS:
                enriched[k] = None

            # Community Type & Residential Subdivision Focus
            is_residential = (
                item.get("incident") == "Nelson"
                or cid_int in [175947, 175948, 175949, 176073]
            )
            enriched["is_residential_focus"] = is_residential
            enriched["community_type"] = (
                "Suburban Residential Subdivision" if is_residential else "Rural WUI Estate / Canyon Parcel"
            )

            enriched["removed_label_conditioned_fields"] = LABEL_CONDITIONED_DEMO_FIELDS
            enriched_items.append(enriched)

        return {
            "total": len(enriched_items),
            "generated_at": data.get("generated_at"),
            "evidence_status": "dev_set_demo",
            "evidence_notes": [
                "These 50 cases are a development set (used to write rules and SFT targets); "
                "held-out evaluation uses output/splits/pilot_test.jsonl.",
                "Fields listed in removed_label_conditioned_fields were generated from the true outcome label and are now always null.",
                "Imagery release WB_2020_R01 is a release date; per-location capture dates were not verified "
                "for these cases (see src/vlm_benchmark/imagery_provenance.py).",
            ],
            "data_sources": {
                "ground_truth": "CAL FIRE Damage Inspection Specialist (DINS) Post-Fire Forensic Database",
                "aerial_imagery": "ESRI Wayback Historical Orthoimagery (WB_2020_R01, WB_2021_R17) & ESRI World Imagery",
                "temporal_epochs": ["2020-02 Pre-Fire", "2021-10 Post-Fire GT", "2024-2026 Current"],
                "fire_incidents": ["Quail Fire (Solano County, CA)", "Apple Fire (Riverside County)", "Nelson Fire (Butte County)"],
                "environmental_forcing": "Red Flag Warning, 25-35 mph gusts, Dead Fuel Moisture (DFM) < 6%, Chaparral/Timber brush",
            },
            "items": enriched_items,
        }

    def get_image_path(self, case_id: str, epoch: str = "prefire") -> Optional[Path]:
        """Get local file path for case image by temporal epoch or streetview."""
        epoch_clean = (epoch or "prefire").lower().strip()
        epoch_file = self.images_dir / f"case_{case_id}_{epoch_clean}.jpg"
        if epoch_file.exists():
            return epoch_file
        if epoch_clean != "prefire":
            return None  # never substitute another epoch's image (audit 2026-10-22)
        p = self.images_dir / f"case_{case_id}.jpg"
        return p if p.exists() else None

    def get_all_results(self) -> Dict[str, Any]:
        """Aggregate evaluation results from all 4 models by case_id."""
        models = [
            ("MiniMax-M3", self.benchmark_dir / "baseline_evaluation_minimax_m3.json"),
            ("claude-haiku-5-5", self.benchmark_dir / "baseline_evaluation_claude_haiku_5_5.json"),
            ("claude-sonnet-5-5", self.benchmark_dir / "baseline_evaluation_claude_sonnet_5_5.json"),
            ("Qwen3.8-27B", self.benchmark_dir / "baseline_evaluation_qwen3_8_27b.json"),
        ]

        by_case: Dict[str, Dict[str, Any]] = {}
        model_summaries: Dict[str, Any] = {}

        for model_name, path in models:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)

            model_summaries[model_name] = {
                "metrics": d.get("metrics", {}),
                "confusion_matrix": d.get("confusion_matrix", {}),
                "total_evaluated": d.get("total_evaluated", 0),
                "evaluated_at": d.get("evaluated_at"),
            }

            for case_rec in d.get("cases", []):
                cid = str(case_rec["case_id"])
                if cid not in by_case:
                    by_case[cid] = {}
                out = dict(case_rec.get("model_output", {}))
                # Never synthesize reasoning text: show only what the model actually returned.
                raw_snippet = out.get("thinking_trace_snippet", "") or ""
                out["thinking_trace"] = raw_snippet
                out["trace_provenance"] = "model_raw_truncated_500" if raw_snippet else "not_captured"
                by_case[cid][model_name] = out

        return {
            "model_summaries": model_summaries,
            "cases": by_case,
        }

    def get_metrics_scoreboard(self) -> Dict[str, Any]:
        """Return standardized scoreboard comparing all 4 baseline models."""
        res = self.get_all_results()
        summaries = res.get("model_summaries", {})
        scoreboard = []

        for name, data in summaries.items():
            m = data.get("metrics", {})
            cm = data.get("confusion_matrix", {})
            scoreboard.append({
                "model_name": name,
                "accuracy": round(m.get("accuracy", 0.0) * 100, 2),
                "precision": round(m.get("precision_destroyed", 0.0) * 100, 2),
                "recall": round(m.get("recall_destroyed", 0.0) * 100, 2),
                "f1_score": round(m.get("f1_score", 0.0), 4),
                "false_alarm_rate": round(m.get("false_alarm_rate_FP", 0.0) * 100, 1),
                "locus_attribution_accuracy": round(m.get("locus_attribution_accuracy", 0.0) * 100, 1),
                "confusion_matrix": cm,
            })

        return {"scoreboard": scoreboard}

    def run_live_prediction(self, model_name: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute on-demand live prediction for any model and case."""
        evaluator = DINSBaselineEvaluator(model_name=model_name)
        prompt = evaluator._format_prompt(case_data)
        res = evaluator.query_model(prompt)

        if not res["success"]:
            return {
                "success": False,
                "error": res.get("error"),
                "latency": res.get("latency", 0),
            }

        parsed = evaluator.parse_prediction(res["text"])
        gt_survival = case_data.get("survival_label", "Unknown")
        gt_locus = case_data.get("ignition_point", "None")

        is_correct = (parsed["predicted_survival"] == gt_survival) if gt_survival != "Unknown" else None
        locus_match = False
        if gt_locus and gt_locus.lower() not in ["none", "unknown"]:
            if gt_locus.lower() in parsed["predicted_vulnerability"].lower():
                locus_match = True

        thinking = parsed.get("thinking_trace") or ""

        return {
            "success": True,
            "model_name": model_name,
            "predicted_survival": parsed["predicted_survival"],
            "predicted_vulnerability": parsed["predicted_vulnerability"],
            "predicted_mechanism": parsed["predicted_mechanism"],
            "is_correct": is_correct,
            "locus_match": locus_match,
            "latency_sec": res["latency"],
            "thinking_trace": thinking,
            "analysis_text": parsed["analysis_text"],
        }
