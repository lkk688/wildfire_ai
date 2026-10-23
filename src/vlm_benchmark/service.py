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
# Written by scripts/data/fetch_dev_case_imagery.py and scripts/data/fetch_dev_case_weather.py
IMAGERY_PROVENANCE_JSON = "dev_case_imagery_provenance.json"
FIRE_WEATHER_JSON = "dev_case_fire_weather.json"
VERIFIED_EPOCHS = ("prefire", "postfire", "current")


# Fields that earlier versions generated from the true outcome label. They are now always None.
LABEL_CONDITIONED_DEMO_FIELDS = [
    "bellwether_prediction", "academic_simulations", "insurance_actuarial",
    "expert_diagnosis", "structured_assessment", "incident_forensics",
]


# How the stored runs in baseline_evaluation_*.json were produced (src/vlm_benchmark/evaluate_baselines.py).
STORED_RUN_NOTES = {
    "input": "text only: DINS building attributes, incident name, county and coordinates; no image",
    "prompt_weather_text": ("the prompt states the same weather for every case ('Red Flag Warning conditions, wind "
                            "gusts 25-35 mph, dead fuel moisture < 6%, surrounding chaparral/timber brush'); this is "
                            "prompt text, not observed weather"),
    "dataset": "50-case development set: 25 destroyed (10 fires) vs 25 survived (all Quail Fire), so fire identity "
               "confounds the label; not a held-out evaluation",
    "trace": "stored responses are truncated to 500 characters",
}


class VLMBenchmarkService:
    def __init__(self, benchmark_dir: Optional[Path] = None):
        self.benchmark_dir = benchmark_dir or BENCHMARK_DIR
        self.images_dir = self.benchmark_dir / "images"
        self.benchmark_json = self.benchmark_dir / "dins_vlm_benchmark_50.json"
        self._cached_catalog = None
        self._cached_results = None
        self._dins_lookup = None

    def _load_json(self, name: str) -> Dict[str, Any]:
        p = self.benchmark_dir / name
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"Error loading {p}: {e}")
            return {}

    def _epoch_entry(self, case_id: str, epoch: str, prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """One imagery epoch, described only by its verified Wayback provenance (never by a fixed date)."""
        names = {"prefire": "Pre-fire", "postfire": "Post-fire", "current": "Latest available"}
        rules = {
            "prefire": "latest Wayback capture before the incident start",
            "postfire": "earliest Wayback capture within 365 days after the incident start",
            "current": "capture served by the newest Wayback release",
        }
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "label": names[epoch],
            "selection_rule": rules[epoch],
            "url": f"/api/vlm/benchmark/image/{case_id}?epoch={epoch}",
            "exists": False,
            "available": False,
            "capture_date": None,
            "unavailable_reason": None,
            "source": "Esri World Imagery Wayback (capture date from the release metadata layer, SRC_DATE)",
        }
        if not prov:
            entry["unavailable_reason"] = "not fetched (run scripts/data/fetch_dev_case_imagery.py)"
            return entry
        f = self.images_dir / f"case_{case_id}_{epoch}_verified.jpg"
        entry.update({
            "capture_date": prov.get("capture_date"),
            "release_date": prov.get("release_date"),
            "release_identifier": prov.get("release_identifier"),
            "release_id": prov.get("release_id"),
            "capture_source": prov.get("capture_source"),
            "capture_provider": prov.get("capture_provider"),
            "capture_resolution_m": prov.get("capture_resolution_m"),
            "days_before_fire": prov.get("staleness_days") if epoch != "current" else None,
            "zoom": prov.get("zoom"),
            "size_px": prov.get("size_px"),
            "image_sha256": prov.get("image_sha256"),
        })
        if prov.get("accepted") and f.exists():
            entry["exists"] = entry["available"] = True
        else:
            entry["unavailable_reason"] = prov.get("reject_reason") or "image file missing"
            entry["url"] = None
        return entry

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
        imagery = self._load_json(IMAGERY_PROVENANCE_JSON).get("cases", {})
        weather = self._load_json(FIRE_WEATHER_JSON).get("cases", {})
        items = data.get("items", [])
        enriched_items = []
        for item in items:
            enriched = dict(item)
            case_id = str(item["id"])
            cid_int = int(item["id"]) if str(item["id"]).isdigit() else None
            dins_props = dins_lookup.get(cid_int, {}) if cid_int is not None else {}

            prov = imagery.get(case_id, {})
            epochs = {ep: self._epoch_entry(case_id, ep, prov.get(ep)) for ep in VERIFIED_EPOCHS}
            enriched["incident_start"] = prov.get("fire_start") or (weather.get(case_id) or {}).get("incident_start")
            enriched["image_url"] = epochs["prefire"]["url"]
            enriched["has_image"] = epochs["prefire"]["available"]

            # APN & Assessor Records
            apn = dins_props.get("APN") or item.get("apn") or "Unknown"
            year_built = item.get("year_built") or dins_props.get("YEARBUILT")
            assessed_val = item.get("assessed_value") or dins_props.get("ASSESSEDIMPROVEDVALUE")

            # Chapter 7A: only the construction year relative to 2008 is known. Compliance itself was never inspected.
            try:
                year_num = int(year_built) if year_built is not None else None
            except (ValueError, TypeError):
                year_num = None
            basis = ("California Building Code Chapter 7A applies to new construction in designated fire hazard "
                     "zones from 2008; whether this parcel was in such a zone, and whether the building complied, "
                     "is not recorded here.")
            if year_num is not None:
                chapter_7a = {
                    "is_compliant": None,
                    "built_after_2008": year_num >= 2008,
                    "code_status": "Built 2008 or later" if year_num >= 2008 else "Built before 2008",
                    "badge_type": "unknown",
                    "cutoff_year": 2008,
                    "standard_summary": basis,
                    "regulatory_basis": "Derived from YEARBUILT only",
                }
            else:
                chapter_7a = {
                    "is_compliant": None,
                    "built_after_2008": None,
                    "code_status": "Year built not recorded",
                    "badge_type": "unknown",
                    "cutoff_year": 2008,
                    "standard_summary": basis,
                    "regulatory_basis": "YEARBUILT missing",
                }

            enriched["assessor_record"] = {
                "apn": apn,
                "year_built": year_built,
                "assessed_improved_value": assessed_val,
                "assessed_value_display": f"${assessed_val:,.0f}" if isinstance(assessed_val, (int, float)) and assessed_val > 0 else "N/A",
                "chapter_7a": chapter_7a,
                "source": "CAL FIRE DINS record (APN, YEARBUILT, ASSESSEDIMPROVEDVALUE)",
            }

            # Multi-temporal imagery: verified Wayback captures only (scripts/data/fetch_dev_case_imagery.py).
            has_streetview = (self.images_dir / f"case_{case_id}_streetview.jpg").exists()
            enriched["multi_epoch_images"] = dict(epochs)
            enriched["multi_epoch_images"]["streetview"] = {
                "epoch": "streetview",
                "url": f"/api/vlm/benchmark/image/{case_id}?epoch=streetview" if has_streetview else None,
                "label": "Street-level photo",
                "capture_date": None,
                "unavailable_reason": None if has_streetview else "no street-level photo stored for this case",
                "source": "Street-level photograph stored with the dev set; capture date not recorded",
                "exists": has_streetview,
                "available": has_streetview,
            }

            # Fire weather: ERA5 via Open-Meteo (scripts/data/fetch_dev_case_weather.py)
            enriched["fire_weather"] = weather.get(case_id) or {
                "available": False,
                "reason": "not fetched (run scripts/data/fetch_dev_case_weather.py)",
                "source": "ERA5 reanalysis via Open-Meteo (may understate local gusts)",
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

            enriched["removed_label_conditioned_fields"] = LABEL_CONDITIONED_DEMO_FIELDS
            enriched_items.append(enriched)

        incidents: Dict[str, Dict[str, Any]] = {}
        for it in enriched_items:
            name = str(it.get("incident") or "").strip()
            rec = incidents.setdefault(name, {"incident": name, "county": it.get("county"),
                                              "start_date": it.get("incident_start"), "cases": 0})
            rec["cases"] += 1
        n_ok = {ep: sum(1 for it in enriched_items if it["multi_epoch_images"][ep]["available"])
                for ep in VERIFIED_EPOCHS}
        return {
            "total": len(enriched_items),
            "generated_at": data.get("generated_at") or data.get("created_at"),
            "evidence_status": "dev_set_demo",
            "evidence_notes": [
                "These 50 cases are a development set (used to write rules and SFT targets); "
                "held-out evaluation uses output/splits/pilot_test.jsonl.",
                "Fields listed in removed_label_conditioned_fields were generated from the true outcome label and are now always null.",
                "Imagery capture dates are read per location from Esri Wayback metadata; an epoch with no verified "
                "capture is marked unavailable instead of being replaced by another image.",
            ],
            "data_sources": {
                "ground_truth": "CAL FIRE Damage Inspection (DINS) records",
                "aerial_imagery": "Esri World Imagery Wayback, per-location capture dates (SRC_DATE)",
                "imagery_coverage": {ep: f"{n}/{len(enriched_items)}" for ep, n in n_ok.items()},
                "fire_weather": "ERA5 reanalysis via Open-Meteo archive API (incident start date + 2 days)",
                "fire_incidents": sorted(incidents.values(), key=lambda r: -r["cases"]),
            },
            "items": enriched_items,
        }

    def get_image_path(self, case_id: str, epoch: str = "prefire") -> Optional[Path]:
        """Local file for a case image. Aerial epochs serve only the verified Wayback crops; nothing is substituted."""
        epoch_clean = (epoch or "prefire").lower().strip()
        if not str(case_id).isdigit():
            return None
        if epoch_clean in VERIFIED_EPOCHS:
            prov = self._load_json(IMAGERY_PROVENANCE_JSON).get("cases", {}).get(str(case_id), {}).get(epoch_clean)
            p = self.images_dir / f"case_{case_id}_{epoch_clean}_verified.jpg"
            return p if prov and prov.get("accepted") and p.exists() else None
        if epoch_clean == "streetview":
            p = self.images_dir / f"case_{case_id}_streetview.jpg"
            return p if p.exists() else None
        return None

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
            "run_notes": STORED_RUN_NOTES,
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
                "total_evaluated": data.get("total_evaluated", 0),
                "evaluated_at": data.get("evaluated_at"),
            })

        return {"scoreboard": scoreboard, "run_notes": STORED_RUN_NOTES}

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
