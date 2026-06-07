"""
CynthAI_v2 — cheater_v8 (terminal-only reward).

Same config as cheater_v7 (Switch A: bounded critic head), but with ALL
dense rewards disabled. Only the terminal signal ±1.0 (win/loss) remains.

Dense rewards zeroed from update 0 via phase schedule:
  dense_phase_breakpoints = (0,)       → breakpoint at update 0
  dense_phase_values      = (0.0, 0.0) → 0.0 before AND after

This is a pure sparse-reward baseline: the agent must learn solely from the
win/loss signal at episode end, with no intermediate KO / HP / count / status
/ hazard shaping. Useful to compare against v7 and measure how much the dense
rewards accelerate learning.

Watch on wandb / metrics.csv:
  - explained_variance   → harder to fit with sparse reward, may be slower
  - move_recall          → proxy for game understanding
  - schedule/dense_scale → should be 0.0 throughout

Usage:
    python run_cheater_v8.py
    python run_cheater_v8.py --resume checkpoints/cheater_v8/agent_000100.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 cheater_v8 (terminal-only reward)")
    parser.add_argument("--resume", default="", help="checkpoint .pt to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        run_name      = "cheater_v8",
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
        value_dump_threshold = 20.0,
        critic_value_bound   = 10.0,   # Switch A — ENABLED
        value_target_clip    = 0.0,    # Switch B — off

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

        # Terminal-only reward — dense scale = 0.0 from update 0
        dense_schedule          = "phase",
        dense_phase_breakpoints = (0,),
        dense_phase_values      = (0.0, 0.0),

        # Checkpointing / eval
        checkpoint_freq = 100,
        eval_freq       = 100,
        eval_n_games    = 500,
        log_every       = 1,
        win_rate_window = 100,

        device = args.device,
    )

    train(cfg)
