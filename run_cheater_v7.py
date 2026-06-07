"""
CynthAI_v2 — cheater_v7 (Switch A: bounded critic head).

Same config as cheater_v6, but with Switch A enabled to fix the diagnosed
critic-output runaway (raw V diverging to hundreds/thousands, ~constant per
update, sign uncorrelated with advantage → dead critic, EV≈0).

  critic_value_bound = 10.0  : critic output squashed to ±10 via tanh
                               (returns are z-scored to std≈1, so 10 ≈ 10σ).
  value_target_clip  = 0.0   : Switch B still OFF — we isolate the effect of A.

Watch on wandb / metrics.csv:
  - critic/vp_max        → should stay ~10 (no more runaway)
  - explained_variance   → should rise above ~0 if the critic starts tracking returns
If EV stays ~0 once bounded, add Switch B (value_target_clip=10.0).

Usage:
    python run_cheater_v7.py
    python run_cheater_v7.py --resume checkpoints/cheater_v7/agent_000100.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 cheater_v7 (Switch A)")
    parser.add_argument("--resume", default="", help="checkpoint .pt to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        run_name      = "cheater_v7",
        resume        = args.resume,
        total_updates = 3000,

        # Rollout
        n_envs    = 64,
        min_steps = 4096,

        # PPO — actor
        n_epochs      = 4,
        batch_size    = 256,
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
        critic_detach          = True,
        critic_n_layers        = 2,
        critic_lr              = 5e-4,
        critic_wd              = 1e-4,
        critic_grad_norm       = 1.0,

        # Critic-stability diagnostics / switches
        value_dump_threshold = 20.0,   # bounded to ±10, so this no longer fires
        critic_value_bound   = 10.0,   # Switch A — ENABLED
        value_target_clip    = 0.0,    # Switch B — off (isolate A)

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
        eval_n_games    = 500,
        log_every       = 1,
        win_rate_window = 100,

        device = args.device,
    )

    train(cfg)
