"""
CynthAI_v2 — Curriculum Max launch script.

3 phases:
  I.   Fondations (0-600)    : mask=0.0, dense=1.0   — full info, full dense rewards
  II.  Transition (600-2500) : mask=0.5, dense=0.5    — medium masking, medium sparse
  III. Maîtrise   (2500-5000): mask=1.0, dense=0.1    — full masking, near-sparse

Usage:
    python run_curriculum_max.py                          # fresh run
    python run_curriculum_max.py --resume checkpoints/curriculum_max_20260510_1234/agent_002000.pt
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 Curriculum Max")
    parser.add_argument("--resume",    default="",    help="checkpoint .pt to resume from")
    parser.add_argument("--run_name",  default="",    help="override auto-generated run name")
    parser.add_argument("--n_envs",    type=int, default=16)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        # Run meta
        run_name        = args.run_name,
        resume          = args.resume,
        total_updates   = 5000,

        # Rollout
        n_envs          = 32,
        min_steps       = 1024,

        # PPO
        n_epochs        = 2,
        batch_size      = 128,
        lr              = 2.5e-4,
        lr_min          = 1e-5,
        warmup_steps    = 20,
        c_value         = 1.0,
        c_entropy       = 0.02,     # doubled — explore more with POMDP masking
        c_pred          = 0.6,      # moderately push prediction heads
        max_grad_norm   = 0.5,
        weight_decay    = 1e-4,

        # Opponent pool
        pool_size               = 30,
        pool_snapshot_threshold = 0.55,
        pool_cooldown           = 5,

        # P1 — POMDP masking (3-phase curriculum)
        mask_schedule           = "phase",
        mask_phase_breakpoints  = (600, 2500),
        mask_phase_values       = (0.0, 0.5, 1.0),

        # P2 — Reward curriculum (3-phase curriculum)
        dense_schedule          = "phase",
        dense_phase_breakpoints = (600, 2500),
        dense_phase_values      = (1.0, 0.5, 0.1),

        # Checkpointing / eval
        checkpoint_freq = 100,
        eval_freq       = 100,
        eval_n_games    = 10,
        log_every       = 1,
        win_rate_window = 100,

        # Device
        device          = args.device,
    )

    train(cfg)