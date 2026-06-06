"""
CynthAI_v2 — cheater_v9 (action-aware critic + probing).

Based on v7 (best run so far), adds:
  - Action-aware critic: cross-attention on action embeddings (critic sees which
    action is being considered, enabling V(s,a) instead of V(s))
  - Inline probing analyses every 3 evals (probe_freq=3)
  - New diagnostics: critic/action_attn_entropy, critic/action_attn_max

Everything else identical to v7:
  - Independent critic with Switch A (value_bound=10.0)
  - Full dense rewards (terminal-only failed catastrophically in v8)
  - 64 envs, 4096 min_steps, batch 256

Usage:
    python run_cheater_v9.py
    python run_cheater_v9.py --resume checkpoints/cheater_v9/agent_000100.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 cheater_v9 (action-aware critic)")
    parser.add_argument("--resume", default="", help="checkpoint .pt to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        run_name      = "cheater_v9b",
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
        use_independent_critic = True,
        critic_n_layers        = 2,
        critic_lr              = 5e-4,
        critic_wd              = 1e-4,
        critic_grad_norm       = 1.0,

        # Action-aware critic — NEW in v9
        critic_action_aware   = True,
        critic_n_cross_layers = 1,
        critic_mask_actions   = True,

        # Victory head
        use_victory_head = True,
        c_victory        = 0.1,

        # Critic-stability switches
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

        # Full dense rewards
        dense_schedule          = "phase",
        dense_phase_breakpoints = (),
        dense_phase_values      = (),

        # Probing — NEW in v9
        probe_freq      = 3,     # every 3 evals = every 300 updates
        probe_min_steps = 2048,

        # Checkpointing / eval
        checkpoint_freq = 100,
        eval_freq       = 100,
        eval_n_games    = 500,
        log_every       = 1,
        win_rate_window = 100,

        device = args.device,
    )

    train(cfg)
