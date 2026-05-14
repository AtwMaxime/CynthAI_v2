"""
CynthAI_v2 — Cheater training with Independent Critic.

Same as run_cheater.py (full info, no POMDP masking) but uses a separate
Transformer for the critic (no shared weights with the actor backbone).

The independent critic has its own lr, weight decay, and grad clip,
all tunable independently from the actor.

Usage:
    python run_cheater_indep_critic.py
    python run_cheater_indep_critic.py --resume checkpoints/cheater_v4/agent_001000.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 Cheater + Independent Critic")
    parser.add_argument("--resume", default="", help="checkpoint .pt to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        run_name      = "cheater_v4",
        resume        = args.resume,
        total_updates = 3000,

        # Rollout
        n_envs    = 32,
        min_steps = 2048,

        # PPO — actor
        n_epochs      = 4,
        batch_size    = 128,
        lr            = 2.5e-4,
        lr_min        = 1e-5,
        warmup_steps  = 20,
        c_value       = 2.0,
        c_entropy     = 0.003,
        c_pred        = 0.6,
        c_attn_entropy = 0.01,
        c_attn_rank    = 0.005,
        max_grad_norm  = 0.5,
        weight_decay   = 1e-4,

        # Independent critic
        use_independent_critic = True,
        critic_n_layers        = 2,
        critic_lr              = 5e-4,   # plus rapide que l'actor
        critic_wd              = 1e-4,
        critic_grad_norm       = 1.0,    # clip plus souple pour le critic

        # Opponent pool
        pool_size               = 10,
        pool_snapshot_freq      = 100,
        pool_snapshot_threshold = 0.55,
        pool_cooldown           = 5,

        # EMA opponent
        ema_decay  = 0.995,
        ema_warmup = 5,

        # No POMDP masking — full information
        mask_schedule          = "phase",
        mask_phase_breakpoints = (),
        mask_phase_values      = (),

        # Full dense rewards
        dense_schedule          = "phase",
        dense_phase_breakpoints = (),
        dense_phase_values      = (),

        # Checkpointing / eval
        checkpoint_freq = 100,
        eval_freq       = 100,
        eval_n_games    = 100,
        log_every       = 1,
        win_rate_window = 100,

        device = args.device,
    )

    train(cfg)
