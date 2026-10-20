#!/usr/bin/env python3
"""
Execution Script: Qwen3.8-27B / Qwen3.5-9B Post-Training with Verifiable Physical Rewards
Usage:
    /home/lkk/miniconda3/envs/grpo/bin/python scripts/run_qwen_grpo_training.py --model Qwen3.8-27B --gpu 0 --mode verify
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_benchmark.qwen_training_pipeline import (
    QwenGRPOVerificationTrainer,
    QwenTrainingConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Qwen WUI Fire Reasoning GRPO Training Runner")
    parser.add_argument("--model", type=str, default="Qwen3.8-27B", choices=["Qwen3.8-27B", "Qwen3.5-9B"],
                        help="Target model architecture")
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1],
                        help="Target GPU device index (0: RTX 6000 102GB, 1: RTX 5090 32GB)")
    parser.add_argument("--mode", type=str, default="verify", choices=["verify", "train"],
                        help="Execution mode (verify: 5-step check; train: full training run)")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    print("=" * 75)
    print("  WildfireReason-GRPO: Qwen Reasoning Post-Training Runner")
    print("=" * 75)
    print(f"  Target Model:      {args.model}")
    print(f"  Target GPU Index:  GPU {args.gpu} ({'RTX 6000 Ada 102GB' if args.gpu == 0 else 'RTX 5090 32GB'})")
    print(f"  Execution Mode:    {args.mode.upper()}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Training Steps:    {args.steps}")
    print("=" * 75)

    config = QwenTrainingConfig(
        model_name_or_path=args.model,
        model_family=args.model,
        target_gpu_index=args.gpu,
        learning_rate=args.lr,
    )

    trainer = QwenGRPOVerificationTrainer(config=config)
    print(f"\n[+] Loaded training dataset from: {config.dataset_path}")

    print("[!] NOTE: this runner calls a SIMULATION of the GRPO objective (no model is loaded,")
    print("    no generation, no backward pass, no optimizer step). It verifies reward/loss wiring only.")
    num_steps = args.steps if args.mode == "train" else 5
    print(f"[+] Launching GRPO verification loop for {num_steps} steps...")
    t0 = time.time()
    res = trainer.run_training_step_simulation(num_steps=num_steps)
    elapsed = round(time.time() - t0, 3)

    if not res.get("success"):
        print(f"[!] Training error: {res.get('error')}")
        sys.exit(1)

    print(f"\n[+] Execution successfully finished in {elapsed}s!")
    print(f"[+] Output log artifact saved to: {res.get('saved_summary')}")
    print("\n--- Training Step Trajectory ---")
    print(f"{'Step':<6}{'Parcel ID':<12}{'GT Status':<12}{'Mean Reward':<14}{'Best Reward':<14}{'GRPO Loss':<12}")
    print("-" * 70)
    for log in res.get("step_logs", []):
        print(
            f"{log['step']:<6}"
            f"#{log['parcel_id']:<11}"
            f"{log['ground_truth_survival']:<12}"
            f"{log['mean_reward']:<14.3f}"
            f"{log['best_reward']:<14.3f}"
            f"{log['grpo_loss']:<12.4f}"
        )
    print("=" * 75)


if __name__ == "__main__":
    import time
    main()
