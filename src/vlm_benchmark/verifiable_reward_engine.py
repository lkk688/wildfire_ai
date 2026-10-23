"""
NOT A VERIFIABLE REWARD (audit 2026-10-22): a keyword-matching heuristic with hard-coded 'physics' rules; it failed
10/10 reliability probes (docs/EVIDENCE_STATUS.md). Do not use it as a training reward.

Verifiable Physical Reward Engine for Wildfire VLM Reasoning (RLVR / GRPO)
Implements verifiable multi-objective physical reward functions to eliminate
hallucinations and prevent decision-making collapse in wildland-urban interface (WUI) scenarios.

Mathematical Formulation:
    R = lambda_1 * R_outcome
      + lambda_2 * R_ignition_locus
      + lambda_3 * R_spatial_grounding
      + lambda_4 * R_physics_consistency
      + lambda_5 * R_cot_format

Physics Consistency Penalty / Rewards:
    - P_anti_physics (-2.0): Hallucinating ember intrusion when Chapter 7A <=1/8" mesh is present.
    - R_wick (+1.0): Correctly identifying attached combustible fences as linear fuel bridges.
    - R_uncertainty (+0.5): Calibrated epistemic uncertainty when rear elevations are occluded.
"""

import re
import json
import math
from typing import Dict, Any, List, Optional, Tuple


class VerifiablePhysicalRewardEngine:
    """Computes verifiable outcome, locus, spatial, and physical conservation rewards."""

    def __init__(
        self,
        lambda_outcome: float = 0.5,
        lambda_locus: float = 1.0,
        lambda_spatial: float = 0.8,
        lambda_physics: float = 1.5,
        lambda_format: float = 0.5,
    ):
        self.l_outcome = lambda_outcome
        self.l_locus = lambda_locus
        self.l_spatial = lambda_spatial
        self.l_physics = lambda_physics
        self.l_format = lambda_format

    def parse_reasoning_output(self, text: str) -> Dict[str, Any]:
        """Parse structured fields and reasoning phases from VLM text output."""
        parsed = {
            "has_think_tags": False,
            "has_phase_1": False,
            "has_phase_2": False,
            "has_phase_3": False,
            "has_phase_4": False,
            "predicted_survival": "Unknown",
            "predicted_locus": "Unknown",
            "predicted_mechanism": "Unknown",
            "detected_attributes": {},
            "identified_wick": False,
            "declared_uncertainty": False,
            "anti_physics_violation": False,
            "text": text,
        }

        # Check think tags and reasoning phases
        if "<think>" in text and "</think>" in text:
            parsed["has_think_tags"] = True

        lower_text = text.lower()
        if "phase 1" in lower_text or "physical attribute inspection" in lower_text:
            parsed["has_phase_1"] = True
        if "phase 2" in lower_text or "environmental forcing" in lower_text:
            parsed["has_phase_2"] = True
        if "phase 3" in lower_text or "causal failure" in lower_text:
            parsed["has_phase_3"] = True
        if "phase 4" in lower_text or "downstream" in lower_text or "parameter synthesis" in lower_text:
            parsed["has_phase_4"] = True

        # Extract survival prediction
        surv_m = re.search(r"PREDICTION:\s*(Destroyed|Survived|No Damage|Affected)", text, re.IGNORECASE)
        if surv_m:
            val = surv_m.group(1).title()
            parsed["predicted_survival"] = "Survived" if val in ["Survived", "No Damage"] else "Destroyed"
        elif "destroyed" in lower_text and "survived" not in lower_text:
            parsed["predicted_survival"] = "Destroyed"
        elif "survived" in lower_text and "destroyed" not in lower_text:
            parsed["predicted_survival"] = "Survived"

        # Extract primary vulnerability / ignition locus
        vuln_m = re.search(r"(?:PRIMARY_VULNERABILITY|IGNITION_LOCUS|WEAKEST_LINK):\s*([^\n\r]+)", text, re.IGNORECASE)
        if vuln_m:
            parsed["predicted_locus"] = vuln_m.group(1).strip()
        else:
            for cand in ["vent", "roof", "eave", "siding", "fence", "deck", "zone 0", "window"]:
                if cand in lower_text:
                    parsed["predicted_locus"] = cand.title()
                    break

        # Check for structural wick identification
        if any(term in lower_text for term in ["combustible fence", "wooden fence", "structural wick", "linear fuel bridge", "fence conduction"]):
            parsed["identified_wick"] = True

        # Check for epistemic uncertainty declaration
        if any(term in lower_text for term in ["rear yard occluded", "unobserved", "elevation not visible", "uncertainty audit", "unverified attic"]):
            parsed["declared_uncertainty"] = True

        return parsed

    def compute_reward(
        self,
        candidate_text: str,
        ground_truth: Dict[str, Any],
        parcel_attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute the multi-objective verifiable reward for a candidate output."""
        attrs = parcel_attributes or {}
        parsed = self.parse_reasoning_output(candidate_text)

        # 1. Outcome Reward R_outcome in [-0.5, +0.5]
        gt_survival = ground_truth.get("survival_label", "Unknown")
        pred_survival = parsed["predicted_survival"]
        if gt_survival in ["Survived", "Destroyed"] and pred_survival in ["Survived", "Destroyed"]:
            r_outcome = 0.5 if pred_survival == gt_survival else -0.5
        else:
            r_outcome = 0.0

        # 2. Ignition Locus Reward R_locus in [0.0, 1.0]
        gt_locus = str(ground_truth.get("ignition_point") or ground_truth.get("WHEREFIRESTARTEDONSTRUCTURE") or "").lower()
        pred_locus = parsed["predicted_locus"].lower()
        if gt_locus and gt_locus not in ["none", "unknown", ""]:
            # Direct or semantic match
            locus_keywords = {
                "vent": ["vent", "screen", "attic", "soffit"],
                "roof": ["roof", "shingle", "tile", "valley"],
                "deck": ["deck", "porch", "patio"],
                "eave": ["eave", "rafter", "soffit"],
                "siding": ["siding", "wall", "cladding"],
                "fence": ["fence", "gate", "wick"],
            }
            matched = False
            for canonical, kws in locus_keywords.items():
                if any(kw in gt_locus for kw in kws):
                    if any(kw in pred_locus for kw in kws) or canonical in pred_locus:
                        matched = True
                        break
            r_locus = 1.0 if matched else 0.0
        else:
            r_locus = 0.5 if pred_locus != "unknown" else 0.0

        # 3. Spatial Grounding Reward R_spatial in [0.0, 1.0]
        # In multi-modal setting, checks presence of spatial zone references (Zone 0, Zone 1, setbacks)
        has_zone0 = "zone 0" in candidate_text.lower() or "5ft" in candidate_text.lower() or "5 ft" in candidate_text.lower()
        has_setback = "separation" in candidate_text.lower() or "buffer" in candidate_text.lower() or "defensible space" in candidate_text.lower()
        r_spatial = (0.5 if has_zone0 else 0.0) + (0.5 if has_setback else 0.0)

        # 4. Physical Law Consistency Reward / Penalty R_physics
        r_physics = 0.0
        details_physics = []

        # Anti-Physics Penalty: Hallucinating attic breach when Chapter 7A mesh (<=1/8") is verified
        vents = str(attrs.get("vent_screen") or ground_truth.get("vent_screen") or "")
        eaves = str(attrs.get("eaves") or ground_truth.get("eaves") or "")
        has_fine_mesh = "<= 1/8" in vents or "1/8" in vents or "chapter 7a" in vents.lower()
        has_enclosed_eaves = "enclosed" in eaves.lower()

        if has_fine_mesh and has_enclosed_eaves:
            # Check if model hallucinated attic ember penetration breach
            if "vent" in pred_locus or "attic ember infiltration" in candidate_text.lower() or "penetrated through vent mesh" in candidate_text.lower():
                r_physics -= 2.0
                parsed["anti_physics_violation"] = True
                details_physics.append("Anti-physics penalty (-2.0): Hallucinated vent ember breach despite verified <=1/8\" mesh.")
            else:
                r_physics += 0.5
                details_physics.append("Physical consistency reward (+0.5): Respected Chapter 7A mesh screening defense.")

        # Combustible Wick Conduction Reward: Attached wooden fence
        fence = str(attrs.get("fence_attached") or ground_truth.get("fence_attached") or "")
        has_attached_fence = "combustible" in fence.lower() or "wood" in fence.lower()
        if has_attached_fence:
            if parsed["identified_wick"]:
                r_physics += 1.0
                details_physics.append("Structural wick reward (+1.0): Correctly identified attached wooden fence as conductive fuel bridge.")
            else:
                r_physics -= 0.5
                details_physics.append("Missed physical wick penalty (-0.5): Failed to identify attached combustible fence.")

        # Epistemic Uncertainty Calibration Reward
        if parsed["declared_uncertainty"]:
            r_physics += 0.5
            details_physics.append("Uncertainty calibration reward (+0.5): Declared unobserved structural angles.")

        # 5. Format & Reasoning Phase Adherence R_cot_format in [-1.0, +0.5]
        format_score = 0.0
        if parsed["has_think_tags"]:
            format_score += 0.1
        if parsed["has_phase_1"]:
            format_score += 0.1
        if parsed["has_phase_2"]:
            format_score += 0.1
        if parsed["has_phase_3"]:
            format_score += 0.1
        if parsed["has_phase_4"]:
            format_score += 0.1

        # Penalize if missing all structured phases
        r_cot_format = format_score if format_score >= 0.3 else -1.0

        # Composite Weighted Reward R
        total_reward = (
            self.l_outcome * r_outcome
            + self.l_locus * r_locus
            + self.l_spatial * r_spatial
            + self.l_physics * r_physics
            + self.l_format * r_cot_format
        )

        return {
            "total_reward": round(total_reward, 4),
            "components": {
                "r_outcome": round(r_outcome, 4),
                "r_locus": round(r_locus, 4),
                "r_spatial": round(r_spatial, 4),
                "r_physics": round(r_physics, 4),
                "r_cot_format": round(r_cot_format, 4),
            },
            "weighted_components": {
                "w_outcome": round(self.l_outcome * r_outcome, 4),
                "w_locus": round(self.l_locus * r_locus, 4),
                "w_spatial": round(self.l_spatial * r_spatial, 4),
                "w_physics": round(self.l_physics * r_physics, 4),
                "w_cot_format": round(self.l_format * r_cot_format, 4),
            },
            "physics_details": details_physics,
            "parsed_features": {
                "predicted_survival": parsed["predicted_survival"],
                "predicted_locus": parsed["predicted_locus"],
                "identified_wick": parsed["identified_wick"],
                "declared_uncertainty": parsed["declared_uncertainty"],
                "anti_physics_violation": parsed["anti_physics_violation"],
                "has_all_phases": all([parsed["has_phase_1"], parsed["has_phase_2"], parsed["has_phase_3"], parsed["has_phase_4"]]),
            },
        }

    def compute_group_advantages(self, group_rewards: List[float], eps: float = 1e-6) -> List[float]:
        """
        Compute Group Relative Policy Optimization (GRPO) advantage:
            A_hat_i = (R_i - mean(R)) / (std(R) + eps)
        """
        if not group_rewards:
            return []
        if len(group_rewards) == 1:
            return [0.0]

        mean_r = sum(group_rewards) / float(len(group_rewards))
        var_r = sum((r - mean_r) ** 2 for r in group_rewards) / float(len(group_rewards))
        std_r = math.sqrt(var_r)

        return [round((r - mean_r) / (std_r + eps), 4) for r in group_rewards]
