"""
Qwen3.8-27B and Qwen3.5-9B Post-Training & Verifiable RLVR Pipeline
Supports LoRA SFT and Group Relative Policy Optimization (GRPO) with
verifiable physical reward functions for wildland-urban interface (WUI) disaster reasoning.

STATUS (Review 1019): `run_training_step_simulation` is a SIMULATION of the GRPO
objective -- no weights are loaded or updated.  Do not report its output as training.

Hardware Target:
- GPU 0: NVIDIA RTX PRO 6000 Blackwell (102.0 GB VRAM)
- GPU 1: NVIDIA GeForce RTX 5090 (33.6 GB VRAM)
"""

import os
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.vlm_benchmark.verifiable_reward_engine import VerifiablePhysicalRewardEngine


@dataclass
class QwenTrainingConfig:
    """Configuration parameters for Qwen3.8-27B and Qwen3.5-9B fine-tuning."""
    model_name_or_path: str = "Qwen3.8-27B"
    model_family: str = "Qwen3.8"  # "Qwen3.8-27B" or "Qwen3.5-9B"
    dataset_path: str = "/Developer/wildfire_ai/SBFire_SJSU-20260730T030001Z-1-001/benchmark_results/w_reason_visual_cot_sft.jsonl"
    output_dir: str = "/Developer/wildfire_ai/output/checkpoints/qwen_wui_reason"
    target_gpu_index: int = 0  # 0 for RTX 6000 102GB, 1 for RTX 5090 32GB
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.05
    group_size_G: int = 4
    kl_coeff: float = 0.04
    temperature: float = 0.7
    max_seq_length: int = 4096


class QwenDatasetLoader:
    """Loads and formats multi-modal Visual-CoT training records."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_samples(self) -> List[Dict[str, Any]]:
        """Read JSONL samples."""
        if not self.dataset_path.exists():
            return []
        samples = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def extract_prompt_and_target(self, sample: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Extract prompt text, gold response, and metadata from chat sample."""
        convs = sample.get("conversations", [])
        user_prompt = ""
        assistant_target = ""
        for c in convs:
            role = c.get("role")
            content = c.get("content")
            if role == "user":
                if isinstance(content, list):
                    texts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    user_prompt = "\n".join(texts)
                else:
                    user_prompt = str(content)
            elif role == "assistant":
                assistant_target = str(content)
        metadata = sample.get("metadata", {})
        metadata["parcel_id"] = sample.get("parcel_id")
        return user_prompt, assistant_target, metadata


class QwenGRPOVerificationTrainer:
    """
    Executes verifiable reward computation, GRPO advantage normalization,
    and loss evaluation for Qwen reasoning policies.
    """

    def __init__(self, config: Optional[QwenTrainingConfig] = None):
        self.config = config or QwenTrainingConfig()
        self.reward_engine = VerifiablePhysicalRewardEngine()
        self.dataset_loader = QwenDatasetLoader(self.config.dataset_path)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_trajectory_group(
        self,
        group_trajectories: List[str],
        ground_truth: Dict[str, Any],
        parcel_attrs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate group of G sampled trajectories:
        Computes multi-objective reward R_i and GRPO normalized advantage A_hat_i.
        """
        G = len(group_trajectories)
        scored_candidates = []
        raw_rewards = []

        for idx, text in enumerate(group_trajectories):
            reward_res = self.reward_engine.compute_reward(text, ground_truth, parcel_attrs)
            total_r = reward_res["total_reward"]
            raw_rewards.append(total_r)
            scored_candidates.append({
                "candidate_index": idx,
                "text": text,
                "reward_details": reward_res,
                "total_reward": total_r,
            })

        # GRPO normalized advantage calculation
        advantages = self.reward_engine.compute_group_advantages(raw_rewards)
        for i, adv in enumerate(advantages):
            scored_candidates[i]["grpo_advantage"] = adv

        mean_r = round(sum(raw_rewards) / float(max(1, G)), 4)
        return {
            "group_size": G,
            "mean_reward": mean_r,
            "raw_rewards": raw_rewards,
            "advantages": advantages,
            "candidates": scored_candidates,
            "best_candidate": max(scored_candidates, key=lambda c: c["total_reward"]),
        }

    def compute_grpo_objective(
        self,
        advantages: List[float],
        logprobs_curr: List[float],
        logprobs_old: List[float],
        logprobs_ref: List[float],
        clip_eps: float = 0.2,
    ) -> Dict[str, float]:
        """
        Calculate Group Relative Policy Optimization (GRPO) clipped surrogate loss:
            J(theta) = E [ min(r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i) - beta * KL ]
        """
        if not advantages or len(advantages) != len(logprobs_curr):
            return {"loss": 0.0, "mean_advantage": 0.0, "mean_kl": 0.0}

        policy_losses = []
        kl_divs = []

        for adv, lp_curr, lp_old, lp_ref in zip(advantages, logprobs_curr, logprobs_old, logprobs_ref):
            # Probability ratio r_i = pi_curr / pi_old
            ratio = math.exp(min(20.0, max(-20.0, lp_curr - lp_old)))
            surr1 = ratio * adv
            surr2 = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps) * adv
            policy_loss = -min(surr1, surr2)
            policy_losses.append(policy_loss)

            # Approximate KL: log(pi_curr / pi_ref)
            kl = lp_curr - lp_ref
            kl_divs.append(max(0.0, kl))

        mean_policy_loss = sum(policy_losses) / float(len(policy_losses))
        mean_kl = sum(kl_divs) / float(len(kl_divs))
        total_loss = mean_policy_loss + self.config.kl_coeff * mean_kl

        return {
            "loss": round(total_loss, 4),
            "policy_surrogate_loss": round(mean_policy_loss, 4),
            "mean_kl": round(mean_kl, 4),
            "mean_advantage": round(sum(advantages) / float(len(advantages)), 4),
        }

    def run_training_step_simulation(self, num_steps: int = 5) -> Dict[str, Any]:
        """
        Execute simulated training steps on real dataset samples to verify
        reward attribution, advantage normalization, and loss dynamics.
        """
        samples = self.dataset_loader.load_samples()
        if not samples:
            return {"success": False, "error": "No training samples found in dataset"}

        step_logs = []
        for step in range(min(num_steps, len(samples))):
            sample = samples[step]
            _, gold_text, meta = self.dataset_loader.extract_prompt_and_target(sample)
            gt = {
                "survival_label": meta.get("ground_truth_survival", "Unknown"),
                "ignition_point": meta.get("ground_truth_locus", "Unknown"),
                "vent_screen": sample.get("vent_screen", "<= 1/8\""),
                "fence_attached": sample.get("fence_attached", "No Fence"),
            }

            # Generate group of 4 candidate variations
            candidates = [
                gold_text,  # Ideal aligned candidate
                # Candidate with missed fence wick
                gold_text.replace("Combustible wooden fence identified", "Fence ignored"),
                # Candidate with anti-physics hallucination
                "PREDICTION: Destroyed\nPRIMARY_VULNERABILITY: Vent\nFAILURE_MECHANISM: Embers\nAttic penetrated through mesh.",
                # Degraded unstructured candidate
                "PREDICTION: Destroyed\nEverything in California burns during Red Flag wind events.",
            ]

            eval_res = self.evaluate_trajectory_group(candidates, gt, gt)

            # Simulated log probabilities for current, old, and ref policies
            logprobs_curr = [-0.15, -0.42, -0.85, -1.25]
            logprobs_old = [-0.25, -0.38, -0.70, -1.10]
            logprobs_ref = [-0.20, -0.35, -0.65, -1.05]

            loss_res = self.compute_grpo_objective(
                eval_res["advantages"], logprobs_curr, logprobs_old, logprobs_ref
            )

            step_logs.append({
                "step": step + 1,
                "parcel_id": meta.get("parcel_id"),
                "ground_truth_survival": gt["survival_label"],
                "mean_reward": eval_res["mean_reward"],
                "best_reward": eval_res["best_candidate"]["total_reward"],
                "grpo_loss": loss_res["loss"],
                "policy_surrogate_loss": loss_res["policy_surrogate_loss"],
                "mean_kl": loss_res["mean_kl"],
                "advantages": eval_res["advantages"],
            })

        # Save verification log
        # Review 1019: no model is loaded, sampled or updated here.  Candidates are the gold text,
        # string edits of it and pre-written wrong answers; log-probs are fixed lists.  This file is
        # an objective/reward wiring check, not evidence of training, convergence or improvement.
        checkpoint_meta = {
            "evidence_status": "simulated_objective_check_no_weights_updated",
            "model_loaded": False,
            "optimizer_steps": 0,
            "model_family": self.config.model_family,
            "simulated_steps": len(step_logs),
            "trained_steps": len(step_logs),  # legacy key: simulated steps, NOT optimizer steps
            "final_mean_reward": step_logs[-1]["mean_reward"] if step_logs else 0.0,
            "final_loss": step_logs[-1]["grpo_loss"] if step_logs else 0.0,
            "step_history": step_logs,
            "hardware_target": "RTX 6000 Ada (102GB) / RTX 5090 (33.6GB)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        save_path = self.output_dir / "qwen_grpo_verification_summary.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_meta, f, indent=2)

        return {
            "success": True,
            "model_family": self.config.model_family,
            "saved_summary": str(save_path),
            "step_logs": step_logs,
        }
