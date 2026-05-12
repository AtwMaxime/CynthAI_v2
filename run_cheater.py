"""
CynthAI_v2 — Cheater (full-info) training.

Trains with NO POMDP masking (mask_ratio=0 always) and full dense rewards.
Goal: validate the architecture works end-to-end when the agent has complete
information about the opponent. If this converges, we know the architecture
is sound and can proceed with:
  - Distillation to a POMDP agent (teacher-student)
  - Asymmetric actor-critic

Usage:
    python run_cheater.py                                # fresh run
    python run_cheater.py --resume checkpoints/cheater_20260512_1200/agent_001000.pt
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 Cheater (full-info) Training")
    parser.add_argument("--resume",    default="",    help="checkpoint .pt to resume from")
    parser.add_argument("--run_name",  default="",    help="override auto-generated run name")
    parser.add_argument("--n_envs",    type=int, default=32)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        # Run meta
        run_name        = f"cheater_{args.run_name}" if args.run_name else "cheater",
        resume          = args.resume,
        total_updates   = 5000,

        # Rollout
        n_envs          = args.n_envs,
        min_steps       = 2048,

        # PPO — cheater: no masking → lower entropy, more epochs
        n_epochs        = 4,
        batch_size      = 128,
        lr              = 2.5e-4,
        lr_min          = 1e-5,
        warmup_steps    = 20,
        c_value         = 2.0,
        c_entropy       = 0.003,     # P16a: réduit pour que policy_loss domine
        c_pred          = 0.6,      # prediction heads still useful for auxiliary signal
        c_attn_entropy  = 0.001,    # P14: cross-attention entropy regularisation
        max_grad_norm   = 0.5,
        weight_decay    = 1e-4,

        # Opponent pool
        pool_size               = 10,
        pool_snapshot_freq      = 100,
        pool_snapshot_threshold = 0.55,
        pool_cooldown           = 5,

        # P10b — EMA opponent
        ema_decay               = 0.995,
        ema_warmup              = 5,

        # POMDP masking — DISABLED (mask_ratio=0 always)
        mask_schedule           = "phase",
        mask_phase_breakpoints  = (),    # empty → compute_mask_ratio returns 0.0
        mask_phase_values       = (),

        # Reward curriculum — DISABLED (full dense rewards always)
        dense_schedule          = "phase",
        dense_phase_breakpoints = (),    # empty → compute_dense_scale returns 1.0
        dense_phase_values      = (),

        # Checkpointing / eval
        checkpoint_freq = 100,
        eval_freq       = 100,
        eval_n_games    = 100,
        log_every       = 1,
        win_rate_window = 100,

        # Device
        device          = args.device,
    )

    train(cfg)